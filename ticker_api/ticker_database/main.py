from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from redis import Redis
from sqlalchemy import create_engine, inspect, text, func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import Session
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
        A class that initializes and manages a KiteConnect connection and Redis client for market data processing.

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

        self.z_connect = ZerodhaConnect(token=token)
        redis_config = {
            "host": redis_host,
            "password": redis_password,
            "port": redis_port,
            "db": redis_db,
            "decode_responses": True,
        }
        self.redis_client = Redis(**redis_config)

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
            logger.info(
                f"td:sync_instruments: successful for  #{num_records_affected} records"
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td:sync_instruments: error while syncing instruments data: {str(e)}"
            )

    @staticmethod
    def _get_sync_details(
        session: Session, instrument_token: int, interval: str
    ) -> Optional[HistoricalDataSyncDetails]:
        return (
            session.query(HistoricalDataSyncDetails)
            .filter_by(instrument_token=instrument_token, interval=interval)
            .first()
        )

    @staticmethod
    def _get_instrument_details(
        session: Session, instrument_token: int
    ) -> Optional[Tuple[str, str, str]]:
        """
        Retrieve instrument details for a given instrument token.

        :param session: SQLAlchemy session
        :param instrument_token: The instrument token to fetch details for
        :return: Tuple of (instrument_type, tradingsymbol, exchange) if found, None otherwise
        """
        instrument = (
            session.query(Instruments)
            .filter_by(instrument_token=instrument_token)
            .first()
        )
        if instrument:
            return (
                instrument.instrument_type,
                instrument.tradingsymbol,
                instrument.exchange,
            )
        return None

    @staticmethod
    def _insert_or_update_historical_data(
        session: Session,
        df: pd.DataFrame,
        interval: str,
        tradingsymbol: str,
        exchange: str,
        continuous: bool,
    ):
        # Prepare the data
        data = [
            {
                "instrument_token": row.instrument_token,
                "tradingsymbol": tradingsymbol,
                "exchange": exchange,
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
        )

        session.execute(stmt)
        session.flush()

    @staticmethod
    def _update_sync_details(
        session: Session,
        instrument_token: int,
        tradingsymbol: str,
        exchange: str,
        interval: str,
        min_date: datetime.date,
        max_date: datetime.date,
        fut_contract_type: str,
    ):
        stmt = insert(HistoricalDataSyncDetails).values(
            instrument_token=instrument_token,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            interval=interval,
            fut_contract_type=fut_contract_type,
            from_date=min_date,
            to_date=max_date,
            status=1,  # Assuming 1 means successfully synced
        )

        stmt = stmt.on_duplicate_key_update(
            tradingsymbol=stmt.inserted.tradingsymbol,
            exchange=stmt.inserted.exchange,
            fut_contract_type=fut_contract_type,
            from_date=stmt.inserted.from_date,
            to_date=stmt.inserted.to_date,
            status=stmt.inserted.status,
        )

        session.execute(stmt)
        session.flush()

    def sync_historical_data(
        self,
        instrument_token: int,
        interval: str,
        fut_contract_type: Optional[str] = None,
    ):
        """
        Synchronize historical data for given instrument_token with database.

        :param instrument_token: The instrument token to fetch data for
        :param interval: The time interval for the data (e.g., 'day', '15minute'. '5minute').
        :param fut_contract_type: Futures contract type identifier (current, mid, far) or None otherwise.
        :return:
        """
        try:
            logger.info(
                f"td:sync_historical_data: starting sync for instrument_token {instrument_token} with interval {interval}"
            )

            # Connect to the specific database
            engine = create_engine(
                self.db_connection_string + self.db_schema_name, echo=False
            )

            with Session(engine) as session:
                instrument_details = self._get_instrument_details(
                    session, instrument_token
                )
                if not instrument_details:
                    logger.error(
                        f"td:sync_historical_data: instrument details not found for token {instrument_token}"
                    )
                    return

                instrument_type, tradingsymbol, exchange = instrument_details
                continuous = instrument_type != "EQ"

                sync_details = self._get_sync_details(
                    session, instrument_token, interval
                )
                if sync_details:
                    from_date = sync_details.to_date
                    to_date = datetime.now().date()
                else:
                    logger.info(
                        f"td:sync_historical_data: no previous sync details found for instrument_token {instrument_token}; starting from 2005-01-01."
                    )
                    from_date = datetime(2005, 1, 1).date()
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
                        interval,
                        tradingsymbol,
                        exchange,
                        continuous,
                    )
                    logger.info(
                        f"td:sync_historical_data: inserted/updated {len(historical_df)} records for instrument_token {instrument_token}"
                    )

                    min_max_dates = (
                        session.query(
                            func.min(HistoricalData.record_date).label("min_date"),
                            func.max(HistoricalData.record_date).label("max_date"),
                        )
                        .filter_by(instrument_token=instrument_token, interval=interval)
                        .first()
                    )

                    if (
                        min_max_dates
                        and min_max_dates.min_date
                        and min_max_dates.max_date
                    ):
                        self._update_sync_details(
                            session,
                            instrument_token,
                            tradingsymbol,
                            exchange,
                            interval,
                            min_max_dates.min_date,
                            min_max_dates.max_date,
                            fut_contract_type,
                        )
                        logger.info(
                            f"td:sync_historical_data: updated sync details for instrument_token {instrument_token}"
                        )
                    else:
                        logger.warning(
                            f"td:sync_historical_data: no data found for instrument_token {instrument_token}"
                        )
                else:
                    logger.warning(
                        f"td:sync_historical_data: no historical data retrieved for instrument_token {instrument_token}"
                    )

                session.commit()

            logger.info(
                f"td:sync_historical_data: sync for instrument_token {instrument_token} with interval {interval} completed"
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td:sync_historical_data: sqlalchemy error syncing historical data for instrument_token {instrument_token}: {str(e)}"
            )
            if "session" in locals():
                session.rollback()
        except Exception as e:
            logger.error(
                f"td:sync_historical_data: unexpected error syncing historical data for instrument_token {instrument_token}: {str(e)}"
            )
            if "session" in locals():
                session.rollback()

    def _create_database_if_not_exists(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(
                    text(f"CREATE DATABASE IF NOT EXISTS {self.db_schema_name}")
                )
                connection.execute(text(f"USE {self.db_schema_name}"))
            logger.info(
                f"td:_create_database_if_not_exists: database '{self.db_schema_name}' created or already exists."
            )
        except SQLAlchemyError as e:
            logger.error(
                f"td:_create_database_if_not_exists: error creating database: {str(e)}"
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
                logger.info("td:create_schema: schema created successfully.")
            else:
                logger.info("td:create_schema: schema already exists, no action taken.")
        except SQLAlchemyError as e:
            logger.error(f"td:create_schema: error creating schema: {str(e)}")

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
                logger.info("td:delete_schema: alembic_version table dropped.")

                # Drop all tables defined in the SQLAlchemy models
                Base.metadata.drop_all(engine)
                logger.info("td:delete_schema: all model-defined tables dropped.")

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
                    f"td:delete_schema: database '{self.db_schema_name}' deleted."
                )

            finally:
                # Ensure the connection is closed
                connection.close()
                engine.dispose()

        except OperationalError as e:
            if "Unknown database" in str(e):
                logger.info(
                    f"td:delete_schema: database '{self.db_schema_name}' does not exist. No action needed."
                )
            else:
                logger.error(f"td:delete_schema: operational error: {str(e)}")
        except SQLAlchemyError as e:
            logger.error(f"td:delete_schema: error deleting schema: {str(e)}")
