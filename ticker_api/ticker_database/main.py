import calendar
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta, TH, WE
from sqlalchemy import create_engine, inspect, text, func, select, and_
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func as sql_func
from zerodha_api import ZerodhaConnect

from ticker_api.settings import (
    get_configuration,
)
from ticker_api.settings import get_logger
from ticker_api.ticker_database.schema import Base
from ticker_api.ticker_database.schema import (
    HistoricalDataSyncDetails,
    HistoricalData,
    Instruments,
    FNOExpiryDates,
)

logger = get_logger()


class TickerDatabase:
    def __init__(
        self,
        token: str,
        redis_host: str = "127.0.0.1",
        redis_password: str = "",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        """
        Provides high level methods to house keep market data using Zerodha API in MySQL database.

        :param token: Encrypted access token required for the Zerodha API.
        :param redis_host: The Redis server hostname or IP address. Defaults to "127.0.0.1".
        :param redis_password: The Redis server password. Defaults to an empty string.
        :param redis_port: The Redis server port. Defaults to 6379.
        :param redis_db: The Redis database number to connect to. Defaults to 0.
        """
        self.config = get_configuration()

        db_username = self.config["db_username"]
        db_password = self.config["db_password"]
        db_host = self.config["db_host"]
        self.db_schema_name = self.config["db_schema_name"]

        # Create SQLAlchemy engine without specifying the database
        self.db_connection_string = (
            f"mysql+pymysql://{db_username}:{db_password}@{db_host}/"
        )
        self.engine = create_engine(
            self.db_connection_string,
            echo=False,
        )
        self.z_connect = ZerodhaConnect(
            token, redis_host, redis_password, redis_port, redis_db
        )

        # Load trading holidays from config
        self.trading_holidays = [
            date.fromisoformat(d) for d in self.config.get("trading_holidays", [])
        ]

    def sync_instruments(self):
        """
        Synchronize instruments data with database.
        :return:
        """
        instruments_df = self.z_connect.instruments()

        try:
            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            # =========================
            # Truncate and Insert
            # =========================
            with engine.connect() as connection:
                connection.execute(
                    text("TRUNCATE TABLE instruments;")
                )  # Truncate the "instruments" table

            # Insert data into the "instruments" table
            with engine.connect() as _:
                num_records_affected = instruments_df.to_sql(
                    "instruments", con=engine, if_exists="append", index=False
                )

            self._update_fno_expiry_dates()

            logger.info(
                f"td::sync_instruments:: successful for  #{num_records_affected} records"
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td::sync_instruments:: error while syncing instruments data: {str(e)}"
            )

    def _update_fno_expiry_dates(self):
        """
        Update FNO expiry dates table with distinct expiry dates from instruments table.
        """
        try:
            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            with Session(engine) as session:
                # Query distinct tuples from instruments table
                stmt = (
                    select(
                        Instruments.exchange,
                        Instruments.segment,
                        Instruments.name,
                        Instruments.instrument_type,
                        Instruments.expiry,
                    )
                    .distinct()
                    .where(
                        and_(
                            Instruments.expiry.isnot(None),
                            Instruments.instrument_type == "FUT",
                        )
                    )
                )

                result = session.execute(stmt)

                # Prepare data for insertion
                data_to_insert = []
                for row in result:
                    expiry_date = row.expiry
                    data_to_insert.append(
                        {
                            "exchange": row.exchange,
                            "segment": row.segment,
                            "name": row.name,
                            "instrument_type": row.instrument_type,
                            "expiry_date": expiry_date,
                            "expiry_weekday": expiry_date.strftime("%A"),
                            "status": 1,
                        }
                    )

                # Perform upsert operation
                stmt = insert(FNOExpiryDates).values(data_to_insert)
                stmt = stmt.on_duplicate_key_update(
                    {
                        "expiry_weekday": stmt.inserted.expiry_weekday,
                        "status": 1,
                        "updated_at": sql_func.now(),  # Set the current timestamp
                    }
                )

                session.execute(stmt)
                session.commit()

            logger.info(
                "td::update_fno_expiry_dates:: fno_expiry_dates updated successfully"
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td::update_fno_expiry_dates:: error while updating fno_expiry_dates: {str(e)}"
            )

    def _is_trading_day(self, check_date):
        """
        Check if the given date is a trading day (not a weekend or holiday).
        """
        return check_date.weekday() < 5 and check_date not in self.trading_holidays

    def _get_last_valid_trading_day(self, year, month, name):
        """
        Get the last valid trading day of the given month and year.

        For BANKNIFTY, use Friday instead of Thursday from July 2023 onwards.
        https://www.moneycontrol.com/news/business/markets/nse-changes-nifty-bank-fo-expiry-day-to-friday-from-thursday-10749221.html
        https://www.thehindubusinessline.com/markets/banknifty-expiry-shifted-to-wednesday-from-september/article67179068.ece
        """
        last_day = date(year, month, calendar.monthrange(year, month)[1])

        if name == "BANKNIFTY" and date(year, month, 1) >= date(2023, 9, 6):
            last_expiry_day = last_day + relativedelta(weekday=WE(-1))
        else:
            last_expiry_day = last_day + relativedelta(weekday=TH(-1))

        while not self._is_trading_day(last_expiry_day):
            last_expiry_day -= timedelta(days=1)

        return last_expiry_day

    def sync_historical_fno_expiry_dates(self, name: str):
        """
        Insert or update all expiry dates since March 1, 2009 into the fno_expiry_dates table
        for a given instrument name, accounting for trading holidays and BANKNIFTY special case.

        :param name: Instrument name
        """
        try:
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            with Session(engine) as session:
                start_date = date(2009, 3, 1)
                end_date = datetime.now().date()
                current_date = start_date

                expiry_dates = []

                while current_date <= end_date:
                    expiry_date = self._get_last_valid_trading_day(
                        current_date.year, current_date.month, name
                    )
                    expiry_dates.append(
                        {
                            "exchange": "NFO",
                            "segment": "NFO-FUT",
                            "name": name,
                            "instrument_type": "FUT",
                            "expiry_date": expiry_date,
                            "expiry_weekday": expiry_date.strftime("%A"),
                            "status": 1,
                        }
                    )
                    current_date += relativedelta(months=1)

                # Perform upsert operation
                insert_stmt = insert(FNOExpiryDates).values(expiry_dates)
                upsert_stmt = insert_stmt.on_duplicate_key_update(
                    expiry_weekday=insert_stmt.inserted.expiry_weekday,
                    status=insert_stmt.inserted.status,
                    updated_at=datetime.now(),
                )
                session.execute(upsert_stmt)
                session.commit()

            logger.info(
                f"td::sync_historical_fno_expiry_dates:: successfully inserted FNO expiry dates for {name}, accounting for trading holidays"
            )

        except SQLAlchemyError as e:
            logger.error(
                f"td::sync_historical_fno_expiry_dates:: error inserting FNO expiry dates for {name}: {str(e)}"
            )

    def get_monthly_expiry_date(
        self,
        exchange: str,
        segment: str,
        name: str,
        expiry: str,
        relative_date: Optional[date] = None,
    ) -> Optional[date]:
        """
        Get the monthly expiry dates based on the given parameters.
        """
        try:

            if expiry.upper() == "NONE":
                logger.info(
                    f"td::get_monthly_expiry_date:: returning `None` since not expiry length specified in the request."
                )
                return None

            attr_str = f"{exchange}|{segment}|{name}|FUT|{expiry}"
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            with Session(engine) as session:
                # Determine the reference date
                ref_date = relative_date if relative_date else datetime.now().date()

                # Query for FUT instrument_type
                query = (
                    select(FNOExpiryDates.expiry_date)
                    .where(
                        FNOExpiryDates.exchange == exchange,
                        FNOExpiryDates.segment == segment,
                        FNOExpiryDates.name == name,
                        FNOExpiryDates.instrument_type == "FUT",
                        FNOExpiryDates.expiry_date >= ref_date,
                    )
                    .order_by(FNOExpiryDates.expiry_date)
                    .limit(3)
                )

                result = session.execute(query).fetchall()

                if not result:
                    logger.warning(
                        f"td::get_monthly_expiry_date:: no expiry dates found for the given parameters: {attr_str}"
                    )
                    return None

                expiry_dates = [row[0] for row in result]

                if len(expiry_dates) < 3:
                    logger.warning(
                        f"td::get_monthly_expiry_date:: less than three expiry dates found for the given parameters: {attr_str}"
                    )
                    return None

                if expiry.upper() == "CURRENT":
                    return expiry_dates[0]
                elif expiry.upper() == "MID":
                    return expiry_dates[1]
                elif expiry.upper() == "FAR":
                    return expiry_dates[2]
                else:
                    logger.error(
                        f"td::get_monthly_expiry_date:: invalid expiry type: {expiry}. Must be CURRENT, MID, or FAR."
                    )
                    return None

        except SQLAlchemyError as e:
            logger.error(
                f"td::get_monthly_expiry_date:: error in get_expiry_date: {str(e)}"
            )
            return None

    @staticmethod
    def _get_historical_sync_details(
        session: Session,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
        interval: str,
    ) -> Optional[HistoricalDataSyncDetails]:
        return (
            session.query(HistoricalDataSyncDetails)
            .filter_by(
                exchange=exchange,
                segment=segment,
                name=name,
                instrument_type=instrument_type,
                expiry=expiry,
                interval=interval,
            )
            .first()
        )

    def _get_instrument_token_from_details(
        self,
        session: Session,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
    ) -> Optional[int]:
        """
        Retrieve instrument token for a given instrument details.
        Returns None if no instrument token is found or if more than one instrument token is found.
        """
        current_exp_date = self.get_monthly_expiry_date(exchange, segment, name, expiry)

        query = select(Instruments.instrument_token).where(
            Instruments.exchange == exchange,
            Instruments.segment == segment,
            Instruments.name == name,
            Instruments.instrument_type == instrument_type,
        )

        # Only add expiry to the query if current_exp_date is not None
        if current_exp_date is not None:
            query = query.where(Instruments.expiry == current_exp_date)

        # Use func.count() to get the number of matching rows
        count_query = select(func.count()).select_from(query.subquery())
        count = session.scalar(count_query)

        if count == 0:
            # No instrument token found
            return None
        elif count > 1:
            # More than one instrument token found
            return None
        else:
            # Exactly one instrument token found
            result = session.execute(query).scalar_one_or_none()
            return result

    @staticmethod
    def _insert_or_update_historical_data(
        session: Session,
        df: pd.DataFrame,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
        interval: str,
        continuous: bool,
    ):
        # Prepare the data
        data = [
            {
                "exchange": exchange,
                "segment": segment,
                "name": name,
                "instrument_type": instrument_type,
                "expiry": expiry,
                "record_datetime": row.record_datetime,
                "record_date": row.record_datetime.date(),
                "record_time": row.record_datetime.time(),
                "interval": interval,
                "open_price": row.open_price,
                "high_price": row.high_price,
                "low_price": row.low_price,
                "close_price": row.close_price,
                "volume": row.volume,
                "oi": row.oi,
                "continuous": int(continuous),
                "status": 1,
            }
            for _, row in df.iterrows()
        ]

        # Perform a batch upsert
        stmt = insert(HistoricalData).values(data)
        stmt = stmt.on_duplicate_key_update(
            open_price=stmt.inserted.open_price,
            high_price=stmt.inserted.high_price,
            low_price=stmt.inserted.low_price,
            close_price=stmt.inserted.close_price,
            volume=stmt.inserted.volume,
            oi=stmt.inserted.oi,
            continuous=stmt.inserted.continuous,
            status=stmt.inserted.status,
            updated_at=sql_func.now(),  # Set the current timestamp
        )

        session.execute(stmt)
        session.flush()

    @staticmethod
    def _update_sync_details(
        session: Session,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
        interval: str,
        min_date: datetime.date,
        max_date: datetime.date,
    ):
        stmt = insert(HistoricalDataSyncDetails).values(
            exchange=exchange,
            segment=segment,
            name=name,
            instrument_type=instrument_type,
            expiry=expiry,
            interval=interval,
            from_date=min_date,
            to_date=max_date,
            status=1,  # Assuming 1 means successfully synced
            updated_at=sql_func.now(),  # Set the current timestamp
        )

        stmt = stmt.on_duplicate_key_update(
            from_date=stmt.inserted.from_date,
            to_date=stmt.inserted.to_date,
            status=stmt.inserted.status,
            updated_at=sql_func.now(),  # Set the current timestamp
        )

        session.execute(stmt)
        session.flush()

    def sync_historical_data(
        self,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
        interval: str,
    ):
        """
        Synchronize historical data for given instrument with database.

        :param exchange: The exchange identifier (e.g. NSE for National Stock Exchange and so on...)
        :param segment: The exchange segment identifier (e.g. NSE, BSE, NFO-FUT, BFO-FUT, MCX-FUT etc.)
        :param name: The name of instrument to sync (e.g. `NIFTY 50', TATAMOTORS etc.)
        :param instrument_type: The type of instrument (e.g. EQ, FUT, CE, PE etc.)
        :param expiry: FnO contract expiry type identifier (CURRENT, MID, FAR) or NONE otherwise.
        :param interval: The time interval for the data (e.g., 'day', '15minute'. '5minute').
        :return:
        """
        attr_str = f"{exchange}|{segment}|{name}|{instrument_type}|{expiry}|{interval}"

        try:
            # Validate expiry parameter
            if expiry not in ["NONE", "CURRENT", "MID", "FAR"]:
                raise ValueError(
                    f"Invalid expiry value: {expiry}. Must be one of 'NONE', 'CURRENT', 'MID', or 'FAR'."
                )
            logger.info(f"td::sync_historical_data:: starting sync for {attr_str}")

            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )
            with Session(engine) as session:
                instrument_token = self._get_instrument_token_from_details(
                    session, exchange, segment, name, instrument_type, expiry
                )
                if not instrument_token:
                    logger.error(
                        f"td::sync_historical_data:: instrument token not found for {attr_str}"
                    )
                    return
                continuous = instrument_type != "EQ"
                logger.info(
                    f"td::sync_historical_data:: instrument token {instrument_token} found for  {attr_str}"
                )
                sync_details = self._get_historical_sync_details(
                    session, exchange, segment, name, instrument_type, expiry, interval
                )
                if sync_details:
                    from_date = sync_details.to_date - timedelta(days=5)
                    to_date = datetime.now().date()
                else:
                    logger.info(
                        f"td::sync_historical_data:: no previous sync details found for {attr_str}: starting from '2009-03-01'"
                    )
                    from_date = datetime(
                        2009, 3, 1
                    ).date()  # `INDIA VIX` is available from here onwards
                    to_date = datetime.now().date()

                historical_df = self.z_connect.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d"),
                    interval=interval,
                    continuous=continuous,
                    oi=True,
                )

                if not historical_df.empty:
                    self._insert_or_update_historical_data(
                        session,
                        historical_df,
                        exchange,
                        segment,
                        name,
                        instrument_type,
                        expiry,
                        interval,
                        continuous,
                    )
                    logger.info(
                        f"td::sync_historical_data:: inserted/updated {len(historical_df)} records for instrument_token {instrument_token}"
                    )

                    min_max_dates = (
                        session.query(
                            func.min(HistoricalData.record_date).label("min_date"),
                            func.max(HistoricalData.record_date).label("max_date"),
                        )
                        .filter_by(
                            exchange=exchange,
                            segment=segment,
                            name=name,
                            instrument_type=instrument_type,
                            expiry=expiry,
                            interval=interval,
                        )
                        .first()
                    )

                    if (
                        min_max_dates
                        and min_max_dates.min_date
                        and min_max_dates.max_date
                    ):
                        self._update_sync_details(
                            session,
                            exchange,
                            segment,
                            name,
                            instrument_type,
                            expiry,
                            interval,
                            min_max_dates.min_date,
                            min_max_dates.max_date,
                        )
                        logger.info(
                            f"td::sync_historical_data:: updated sync details for {attr_str}"
                        )
                    else:
                        logger.warning(
                            f"td::sync_historical_data:: no data found for {attr_str} in historical_data table"
                        )
                else:
                    logger.warning(
                        f"td::sync_historical_data:: no historical data retrieved for {attr_str}"
                    )

                session.commit()

            logger.info(f"td::sync_historical_data:: sync for {attr_str} completed.")
        except SQLAlchemyError as e:
            logger.error(
                f"td::sync_historical_data:: sqlalchemy error syncing historical data for {attr_str}: {str(e)}"
            )
            if "session" in locals():
                session.rollback()
        except Exception as e:
            logger.error(
                f"td::sync_historical_data:: unexpected error syncing historical data for {attr_str}: {str(e)}"
            )
            if "session" in locals():
                session.rollback()

    def sync_historical_data_all(self):
        """
        Synchronize historical data for all instruments and intervals in the historical_data_sync_details table.

        :return:
        """
        logger.info(
            "td::sync_historical_data_all:: starting synchronization for all historical data"
        )

        try:
            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            with Session(engine) as session:
                # Step 1: Select all tuples from historical_data_sync_details
                sync_details = session.query(
                    HistoricalDataSyncDetails.exchange,
                    HistoricalDataSyncDetails.segment,
                    HistoricalDataSyncDetails.name,
                    HistoricalDataSyncDetails.instrument_type,
                    HistoricalDataSyncDetails.expiry,
                    HistoricalDataSyncDetails.interval,
                ).all()

                total_instruments = len(sync_details)
                logger.info(
                    f"td::sync_historical_data_all:: found {total_instruments} instruments to synchronize"
                )

                # Step 2: Iterate through the tuples and call sync_historical_data
                for i, (
                    exchange,
                    segment,
                    name,
                    instrument_type,
                    expiry,
                    interval,
                ) in enumerate(sync_details, 1):
                    try:
                        attr_str = f"{exchange}|{segment}|{name}|{instrument_type}|{expiry}|{interval}"
                        logger.info(
                            f"td::sync_historical_data_all:: syncing instrument {i}/{total_instruments} - {attr_str}"
                        )

                        self.sync_historical_data(
                            exchange=exchange,
                            segment=segment,
                            name=name,
                            instrument_type=instrument_type,
                            expiry=expiry,
                            interval=interval,
                        )
                    except Exception as e:
                        logger.error(
                            f"td::sync_historical_data_all:: error syncing data for {attr_str}: {str(e)}"
                        )

                logger.info(
                    "td::sync_historical_data_all: completed synchronization for all historical data"
                )

        except SQLAlchemyError as e:
            logger.error(
                f"td::sync_historical_data_all:: sqlalchemy error during synchronization: {str(e)}"
            )
        except Exception as e:
            logger.error(
                f"td::sync_historical_data_all:: unexpected error during synchronization: {str(e)}"
            )
        logger.info(
            "td::sync_historical_data_all:: finished historical data synchronization process"
        )

    def _create_database_if_not_exists(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text(f"CREATE DATABASE IF NOT EXISTS {self.db_schema_name}")
                )
                connection.execute(text(f"USE {self.db_schema_name}"))
            logger.info(
                f"td::_create_database_if_not_exists: database '{self.db_schema_name}' created or already exists."
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td::_create_database_if_not_exists: error creating database: {str(e)}"
            )
            raise

    def create_schema(self):
        """
        Create database schema if it doesn't exist.
        :return:
        """
        try:
            # First, ensure the database exists
            self._create_database_if_not_exists()

            # Recreate the engine with the specific database
            self.engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            if not existing_tables:
                Base.metadata.create_all(self.engine)
                logger.info("td::create_schema: schema created successfully.")
            else:
                logger.info(
                    "td::create_schema: schema already exists, no action taken."
                )
        except SQLAlchemyError as e:
            logger.error(f"td::create_schema: error creating schema: {str(e)}")

    def delete_schema(self):
        """
        Deletes database schema if it exists.

        :return:
        """
        try:
            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            # Create a connection that we'll use for raw SQL execution
            connection = engine.connect()

            try:
                # Drop the alembic_version table if it exists
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
                logger.info("td::delete_schema: alembic_version table dropped.")

                # Drop all tables defined in the SQLAlchemy models
                Base.metadata.drop_all(engine)
                logger.info("td::delete_schema: all model-defined tables dropped.")

                # Close the connection to the specific database
                connection.close()
                engine.dispose()

                # Connect to the MySQL server without specifying a database
                engine = create_engine(self.db_connection_string, echo=False)
                connection = engine.connect()

                # Drop the entire database
                connection.execute(
                    text(f"DROP DATABASE IF EXISTS {self.db_schema_name}")
                )
                logger.info(
                    f"td::delete_schema: database '{self.db_schema_name}' deleted."
                )

            finally:
                # Ensure the connection is closed
                connection.close()
                engine.dispose()

        except OperationalError as e:
            if "Unknown database" in str(e):
                logger.info(
                    f"td::delete_schema: database '{self.db_schema_name}' does not exist. No action needed."
                )
            else:
                logger.error(f"td::delete_schema: operational error: {str(e)}")
        except SQLAlchemyError as e:
            logger.error(f"td::delete_schema: error deleting schema: {str(e)}")
