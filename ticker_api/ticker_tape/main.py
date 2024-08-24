import calendar
import re
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any

import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta, TH, WE, FR, MO
from sqlalchemy import select, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from ticker_api.exceptions import InvalidExchangeException, InvalidSegmentException
from ticker_api.settings import get_configuration, get_logger
from ticker_api.ticker_database.schema import Instruments, HistoricalData

logger = get_logger()


class TickerTape:
    SUPPORTED_EXCHANGES = {"NSE", "BSE", "NFO", "BFO"}
    MAPPING_INDICES_TO_FO_SYMBOLS = {
        ("NIFTY 50", "NSE"): ("NIFTY", "Thursday"),
        ("NIFTY BANK", "NSE"): ("BANKNIFTY", "Wednesday"),
        # https://blog.dhan.co/sensex-and-bankex-fo-expiry-days/
        ("SENSEX", "BSE"): ("SENSEX", "Friday"),
        ("BANKEX", "BSE"): ("BANKEX", "Monday"),
        # Add other symbols here if needed, with custom day as the default
        # ('SOME_OTHER_SYMBOL', 'EXCHANGE'): ('OTHER_SYMBOL', 'EXPIRY_DAY'),
        # TODO: CDS compatibility based on,
        #  https://www.nseindia.com/products-services/currency-derivatives-contract-specification-inr
    }
    MAPPING_FO_SYMBOLS_TO_INDICES = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}
    MAPPING_EQUITY_TO_FO_EXCHANGE = {"NSE": "NFO", "BSE": "BFO"}

    def __init__(self):
        """
        Provides high level methods to interact with market data stored in MySQL database.
        """
        self.config = get_configuration()

        db_username = self.config["db_username"]
        db_password = self.config["db_password"]
        db_host = self.config["db_host"]
        self.db_schema_name = self.config["db_schema_name"]

        # Create SQLAlchemy engine without specifying the database
        self.db_connection_string = f"mysql+pymysql://{db_username}:{db_password}@{db_host}/{self.db_schema_name}"
        self.engine = create_engine(
            self.db_connection_string,
            echo=False,
        )
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def _extract_spot_name(tradingsymbol: str) -> str:
        """
        Extract the spot name from the derivatives contract name.

        E.g.
            NIFTY2481423900PE -> NIFTY
            TCS24AUG24550PE -> TCS
            EURINR24AUG88.5PE -> EURINR
            BANKNIFTY24AUGFUT -> BANKNIFTY
            EICHERMOT -> EICHERMOT
        :param tradingsymbol:
        :return:
        """
        # Pattern for futures and options
        pattern = r"^((?:[A-Z]+\s?)+)(?:\d{2}(?:[A-Z]{3}|[1-9OND])(?:\d{2,}(?:\.\d+)?[CP]E|FUT))?$"
        match = re.match(pattern, tradingsymbol)
        if match:
            return match.group(1)

        return tradingsymbol  # If no pattern matches, return the original symbol

    @contextmanager
    def _session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _get_next_three_active_contracts(
        spot_name: str, expiry_day: str, timezone: str
    ) -> list:
        # Set up the expiry day mapping
        expiry_day_map = {"Monday": MO, "Wednesday": WE, "Thursday": TH, "Friday": FR}
        expiry_weekday = expiry_day_map.get(
            expiry_day, TH
        )  # Default to Thursday if invalid day provided

        # Get current date in the specified timezone
        current_date = datetime.now(pytz.timezone(timezone)).date()

        # Function to get the last trading day of a given month
        def get_last_trading_day(year, month):
            last_day = date(year, month, calendar.monthrange(year, month)[1])
            return last_day + relativedelta(weekday=expiry_weekday(-1))

        # Get the expiry date of the current month
        current_expiry = get_last_trading_day(current_date.year, current_date.month)

        # Determine the start month for the next three active contracts
        if current_date > current_expiry:
            start_date = current_date + relativedelta(months=1)
        else:
            start_date = current_date

        # Generate the next three active contract names
        contract_names = []
        for i in range(3):
            contract_date = start_date + relativedelta(months=i)
            contract_name = f'{spot_name}{contract_date.strftime("%y%b")}FUT'.upper()
            contract_names.append(contract_name)

        return contract_names

    def _get_details_by_symbol(
        self,
        tradingsymbol: str,
        exchange: str,
        return_db_details: bool,
        fetch_futures: bool,
    ) -> Dict[str, Any]:
        try:
            with self._session_scope() as session:
                # Construct the query to get the instrument details
                query = select(Instruments).where(
                    (Instruments.tradingsymbol == tradingsymbol)
                    & (Instruments.exchange == exchange)
                )

                # Execute the query
                result = session.execute(query).scalar_one_or_none()

                if result is None:
                    return {}  # Return an empty dict if no matching instrument is found

                exclude_keys = {"id", "status", "updated_at", "created_at"}
                result_dict = {
                    column.name: getattr(result, column.name)
                    for column in Instruments.__table__.columns
                    if return_db_details or column.name not in exclude_keys
                }
                for key, value in result_dict.items():
                    if isinstance(value, Decimal):
                        result_dict[key] = float(value)

                if fetch_futures:
                    mapped_symbol, expiry_day = self.MAPPING_INDICES_TO_FO_SYMBOLS.get(
                        (tradingsymbol, exchange), (tradingsymbol, "Thursday")
                    )
                    futures_contract_names = self._get_next_three_active_contracts(
                        spot_name=self._extract_spot_name(mapped_symbol),
                        expiry_day=expiry_day,
                        timezone=self.config["timezone"],
                    )

                    futures_contracts = []
                    for future_symbol in futures_contract_names:
                        mapped_exchange = self.MAPPING_EQUITY_TO_FO_EXCHANGE.get(
                            exchange, exchange
                        )
                        future_details = self.get_details(
                            future_symbol,
                            mapped_exchange,
                            return_db_details,
                        )
                        if future_details:
                            futures_contracts.append(future_details)
                    result_dict["futures_contracts"] = futures_contracts

                return result_dict

        except SQLAlchemyError as e:
            logger.info(
                f"tt:_get_details_by_symbol: An error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception

    def _get_details_by_token(
        self, instrument_token: int, return_db_details: bool
    ) -> Dict[str, Any]:
        try:
            with self._session_scope() as session:
                # Query the Instruments table by instrument_token
                result = session.execute(
                    select(Instruments).where(
                        Instruments.instrument_token == instrument_token
                    )
                ).scalar_one_or_none()

                if result is None:
                    return {}

                tradingsymbol = result.tradingsymbol
                exchange = result.exchange
                if exchange not in self.SUPPORTED_EXCHANGES:
                    raise InvalidExchangeException(
                        f"tt:_get_details_by_token: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                    )

                # Determine whether to fetch futures based on the tradingsymbol
                fetch_futures = not tradingsymbol.endswith("FUT")

                # Use _get_details_by_symbol to get all details including futures
                return self._get_details_by_symbol(
                    tradingsymbol, exchange, return_db_details, fetch_futures
                )
        except SQLAlchemyError as e:
            logger.info(
                f"tt:_get_details_by_token: An error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception

    def get_details(
        self,
        tradingsymbol: Optional[str] = None,
        exchange: Optional[str] = None,
        instrument_token: Optional[int] = None,
        return_db_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Get details of an instrument.

        :param tradingsymbol: The trading symbol of the instrument.
        :param exchange: The exchange where the instrument is traded.
        :param instrument_token: The unique token identifying the instrument.
        :param return_db_details: Whether to return additional database details.
        :return: A dictionary containing the instrument details.
        """
        if tradingsymbol is not None and exchange is not None:
            if exchange not in self.SUPPORTED_EXCHANGES:
                raise InvalidExchangeException(
                    f"tt:get_details: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                )
            fetch_futures = not tradingsymbol.endswith("FUT")
            return self._get_details_by_symbol(
                tradingsymbol, exchange, return_db_details, fetch_futures
            )
        elif instrument_token is not None:
            return self._get_details_by_token(instrument_token, return_db_details)
        else:
            raise ValueError(
                "tt:get_details: invalid parameters! please provide either (tradingsymbol, exchange) or instrument_token"
            )

    @staticmethod
    def _get_segment_category(
        session: Session, tradingsymbol: str, exchange: str
    ) -> str:
        instrument = session.execute(
            select(Instruments).where(
                (Instruments.tradingsymbol == tradingsymbol)
                & (Instruments.exchange == exchange)
            )
        ).scalar_one_or_none()

        if not instrument:
            raise ValueError(
                f"tt:_get_segment_category: instrument not found for {tradingsymbol} on {exchange}"
            )

        segment = instrument.segment
        if segment in ("NSE", "BSE"):
            return "EQUITY"
        elif segment in ("BFO-FUT", "NFO-FUT"):
            return "FUTURES"
        elif segment == "INDICES":
            return "INDEX"
        else:
            raise InvalidSegmentException(
                f"tt:_get_segment_category: invalid segment: {segment}"
            )

    @staticmethod
    def _get_historical_data(
        session: Session, instrument_token: int, interval: str
    ) -> pd.DataFrame:
        query = select(
            HistoricalData.instrument_token,
            HistoricalData.tradingsymbol,
            HistoricalData.exchange,
            HistoricalData.record_date,
            HistoricalData.record_datetime,
            HistoricalData.record_time,
            HistoricalData.open_price,
            HistoricalData.high_price,
            HistoricalData.low_price,
            HistoricalData.close_price,
            HistoricalData.volume,
            HistoricalData.oi,
        ).where(
            (HistoricalData.instrument_token == instrument_token)
            & (HistoricalData.interval == interval)
            & (HistoricalData.status == 1)
        )

        result = session.execute(query)

        # Get column names
        columns = result.keys()

        # Fetch all rows and create a list of dictionaries
        data = [dict(zip(columns, row)) for row in result.fetchall()]

        # Create DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(
                f"tt:_get_historical_data: no data found for instrument_token {instrument_token} with interval {interval}"
            )
            return df

        # Convert price columns to float
        price_columns = ["open_price", "high_price", "low_price", "close_price"]
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert volume and oi to int
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").astype("Int64")

        # Set index
        if interval == "day":
            df.set_index("record_date", inplace=True)
        else:
            df.set_index("record_datetime", inplace=True)

        return df

    def _get_vix_data(self, session: Session, interval: str) -> pd.DataFrame:
        vix_instrument = session.execute(
            select(Instruments).where(
                (Instruments.tradingsymbol == "INDIA VIX")
                & (Instruments.exchange == "NSE")
            )
        ).scalar_one_or_none()

        if not vix_instrument:
            logger.warning("tt:_get_vix_data: india vix instrument not found")
            return pd.DataFrame()

        logger.info(
            f"tt:_get_vix_data: querying for vix data of INDIA VIX@NSE with interval {interval}"
        )
        df = self._get_historical_data(
            session, vix_instrument.instrument_token, interval
        )

        if df.empty:
            return df

        # Rename and select only VIX-related columns
        vix_columns = {
            "open_price": "vix_open",
            "high_price": "vix_high",
            "low_price": "vix_low",
            "close_price": "vix_close",
        }
        df = df.rename(columns=vix_columns)[list(vix_columns.values())]

        # Ensure index is unique
        if df.index.duplicated().any():
            logger.warning(
                "tt:_get_vix_data: duplicate index values found. keeping last occurrence."
            )
            df = df[~df.index.duplicated(keep="last")]

        return df

    def _get_index_fut_data(
        self, session: Session, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        mapped_symbol, expiry_day = self.MAPPING_INDICES_TO_FO_SYMBOLS.get(
            (tradingsymbol, exchange), (tradingsymbol, "Thursday")
        )
        future_symbols = self._get_next_three_active_contracts(
            spot_name=self._extract_spot_name(mapped_symbol),
            expiry_day=expiry_day,
            timezone=self.config["timezone"],
        )[:2]

        dfs = []
        for i, future_symbol in enumerate(future_symbols):
            mapped_exchange = self.MAPPING_EQUITY_TO_FO_EXCHANGE.get(exchange, exchange)
            future_instrument = session.execute(
                select(Instruments).where(
                    (Instruments.tradingsymbol == future_symbol)
                    & (Instruments.exchange == mapped_exchange)
                )
            ).scalar_one_or_none()

            if future_instrument:
                logger.info(
                    f"tt:_get_futures_data: querying for index futures data of {future_symbol}@{mapped_exchange} with interval {interval}"
                )
                df = self._get_historical_data(
                    session, future_instrument.instrument_token, interval
                )

                if df.empty:
                    continue

                if i == 0:
                    df = df[
                        [
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                            "volume",
                            "oi",
                        ]
                    ]
                else:
                    df = df[["oi"]].rename(columns={"oi": "oi_next"})
                dfs.append(df)

        if not dfs:
            logger.warning(
                f"tt:_get_futures_data: no index futures data found for {tradingsymbol} on {exchange}"
            )
            return pd.DataFrame()

        combined_df = pd.concat(dfs, axis=1)

        # Ensure index is unique
        if combined_df.index.duplicated().any():
            logger.warning(
                "tt:_get_futures_data: duplicate index values found. keeping last occurrence."
            )
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        # Sum the oi columns
        oi_columns = combined_df.filter(like="oi").columns
        combined_df["oi"] = combined_df[oi_columns].sum(axis=1)

        # Rename columns with 'fut_' prefix and select only relevant columns
        fut_columns = {
            "open_price": "index_open",
            "high_price": "index_high",
            "low_price": "index_low",
            "close_price": "index_close",
            "volume": "index_volume",
            "oi": "index_oi",
        }
        renamed_df = combined_df.rename(columns=fut_columns)[list(fut_columns.values())]
        return renamed_df

    def _get_futures_data(
        self, session: Session, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        mapped_symbol, expiry_day = self.MAPPING_INDICES_TO_FO_SYMBOLS.get(
            (tradingsymbol, exchange), (tradingsymbol, "Thursday")
        )
        future_symbols = self._get_next_three_active_contracts(
            spot_name=self._extract_spot_name(mapped_symbol),
            expiry_day=expiry_day,
            timezone=self.config["timezone"],
        )[:2]

        dfs = []
        for i, future_symbol in enumerate(future_symbols):
            mapped_exchange = self.MAPPING_EQUITY_TO_FO_EXCHANGE.get(exchange, exchange)
            future_instrument = session.execute(
                select(Instruments).where(
                    (Instruments.tradingsymbol == future_symbol)
                    & (Instruments.exchange == mapped_exchange)
                )
            ).scalar_one_or_none()

            if future_instrument:
                logger.info(
                    f"tt:_get_futures_data: querying for futures data of {future_symbol}@{mapped_exchange} with interval {interval}"
                )
                df = self._get_historical_data(
                    session, future_instrument.instrument_token, interval
                )

                if df.empty:
                    continue

                if i == 0:
                    df = df[
                        [
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                            "volume",
                            "oi",
                        ]
                    ]
                else:
                    df = df[["oi"]].rename(columns={"oi": "oi_next"})
                dfs.append(df)

        if not dfs:
            logger.warning(
                f"tt:_get_futures_data: no futures data found for {tradingsymbol} on {exchange}"
            )
            return pd.DataFrame()

        combined_df = pd.concat(dfs, axis=1)

        # Ensure index is unique
        if combined_df.index.duplicated().any():
            logger.warning(
                "tt:_get_futures_data: duplicate index values found. keeping last occurrence."
            )
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        # Sum the oi columns
        oi_columns = combined_df.filter(like="oi").columns
        combined_df["oi"] = combined_df[oi_columns].sum(axis=1)

        # Rename columns with 'fut_' prefix and select only relevant columns
        fut_columns = {
            "open_price": "fut_open",
            "high_price": "fut_high",
            "low_price": "fut_low",
            "close_price": "fut_close",
            "volume": "fut_volume",
            "oi": "fut_oi",
        }
        renamed_df = combined_df.rename(columns=fut_columns)[list(fut_columns.values())]
        return renamed_df

    def _get_equity_data(
        self, session: Session, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        logger.info(
            f"tt:_get_equity_data: querying for equity data of {tradingsymbol}@{exchange} with interval {interval}"
        )
        instrument = session.execute(
            select(Instruments).where(
                (Instruments.tradingsymbol == tradingsymbol)
                & (Instruments.exchange == exchange)
            )
        ).scalar_one_or_none()

        if not instrument:
            logger.warning(
                f"tt:_get_equity_data: equity instrument not found for {tradingsymbol} on {exchange}"
            )
            return pd.DataFrame()

        df = self._get_historical_data(session, instrument.instrument_token, interval)

        if df.empty:
            return df

        # Rename and select only EQUITY-related columns
        equity_columns = {
            "open_price": "spot_open",
            "high_price": "spot_high",
            "low_price": "spot_low",
            "close_price": "spot_close",
            "volume": "spot_volume",
        }
        df = df.rename(columns=equity_columns)[list(equity_columns.values())]

        # Ensure index is unique
        if df.index.duplicated().any():
            logger.warning(
                "tt:_get_equity_data: duplicate index values found. keeping last occurrence."
            )
            df = df[~df.index.duplicated(keep="last")]

        return df

    def _get_data_for_x_by_symbol(
        self, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        try:
            with self._session_scope() as session:
                category = self._get_segment_category(session, tradingsymbol, exchange)

                vix_data_df = self._get_vix_data(session, interval)

                if category == "INDEX":
                    if tradingsymbol == "INDIA VIX":
                        return vix_data_df
                    else:
                        index_tradingsymbol = "NIFTY 50"
                        index_tradingsymbol = (
                            "NIFTY BANK"
                            if tradingsymbol == "NIFTY 50"
                            else index_tradingsymbol
                        )
                        index_fut_data_df = self._get_index_fut_data(
                            session, index_tradingsymbol, exchange, interval
                        )
                        fut_data_df = self._get_futures_data(
                            session, tradingsymbol, exchange, interval
                        )
                        eq_data_df = self._get_equity_data(
                            session, tradingsymbol, exchange, interval
                        )
                        return pd.concat(
                            [eq_data_df, index_fut_data_df, fut_data_df, vix_data_df],
                            axis=1,
                        )
                elif category == "FUTURES":
                    if exchange not in ["NFO"]:
                        raise InvalidExchangeException(
                            f"tt:_get_data_for_x_by_symbol: exchange {exchange} is not supported; only NFO are allowed."
                        )

                    index_tradingsymbol = "NIFTY 50"
                    index_exchange = "NSE"
                    index_fut_data_df = self._get_index_fut_data(
                        session, index_tradingsymbol, index_exchange, interval
                    )
                    fut_data_df = self._get_futures_data(
                        session, tradingsymbol, exchange, interval
                    )

                    equity_exchange = "NSE"
                    equity_tradingsymbol = self.MAPPING_FO_SYMBOLS_TO_INDICES.get(
                        self._extract_spot_name(tradingsymbol), tradingsymbol
                    )

                    eq_data_df = self._get_equity_data(
                        session, equity_tradingsymbol, equity_exchange, interval
                    )
                    return pd.concat(
                        [eq_data_df, index_fut_data_df, fut_data_df, vix_data_df],
                        axis=1,
                    )
                elif category == "EQUITY":
                    if exchange not in ["NSE"]:
                        raise InvalidExchangeException(
                            f"tt:_get_data_for_x_by_symbol: exchange {exchange} is not supported; only NSE are allowed."
                        )

                    index_tradingsymbol = "NIFTY 50"
                    index_fut_data_df = self._get_index_fut_data(
                        session, index_tradingsymbol, exchange, interval
                    )
                    fut_data_df = self._get_futures_data(
                        session, tradingsymbol, exchange, interval
                    )
                    eq_data_df = self._get_equity_data(
                        session, tradingsymbol, exchange, interval
                    )
                    return pd.concat(
                        [eq_data_df, index_fut_data_df, fut_data_df, vix_data_df],
                        axis=1,
                    )

                else:
                    raise InvalidSegmentException(
                        f"tt:_get_data_for_x_by_symbol: invalid category: {category}"
                    )

        except SQLAlchemyError as e:
            logger.error(
                f"tt:_get_data_for_x_by_symbol:an error occurred in sqlalchemy operations: {str(e)}"
            )
            return pd.DataFrame()

    def _get_data_for_x_by_token(
        self, instrument_token: int, interval: str
    ) -> pd.DataFrame:
        try:
            with self._session_scope() as session:
                # Query the Instruments table by instrument_token
                result = session.execute(
                    select(Instruments).where(
                        Instruments.instrument_token == instrument_token
                    )
                ).scalar_one_or_none()

                if result is None:
                    return pd.DataFrame()

                tradingsymbol = result.tradingsymbol
                exchange = result.exchange
                if exchange not in self.SUPPORTED_EXCHANGES:
                    raise InvalidExchangeException(
                        f"tt:_get_data_for_x_by_token: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                    )

                # Use _get_data_for_x_by_symbol to get all data
                return self._get_data_for_x_by_symbol(tradingsymbol, exchange, interval)
        except SQLAlchemyError as e:
            logger.info(
                f"tt:_get_data_for_x_by_token: an error occurred while fetching data for x: {str(e)}"
            )
            return (
                pd.DataFrame()
            )  # Return an empty pd.DataFrame in case of any exception

    def get_data_for_x(
        self,
        tradingsymbol: Optional[str] = None,
        exchange: Optional[str] = None,
        instrument_token: Optional[int] = None,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Returns combined pd.DataFrame of equity and/or indices, with futures and vix with date index

        :param tradingsymbol: The trading symbol of the instrument.
        :param exchange: The exchange where the instrument is traded.
        :param instrument_token: The unique token identifying the instrument.
        :param interval: The interval of historical data.
        :return: pd.DataFrame
        """
        try:
            if tradingsymbol and exchange:
                if exchange not in self.SUPPORTED_EXCHANGES:
                    raise InvalidExchangeException(
                        f"tt:get_data_for_x: Invalid exchange {exchange}"
                    )
                df = self._get_data_for_x_by_symbol(tradingsymbol, exchange, interval)
            elif instrument_token:
                df = self._get_data_for_x_by_token(instrument_token, interval)
            else:
                raise ValueError(
                    "tt:get_data_for_x: invalid parameters! provide either (tradingsymbol, exchange) or instrument_token"
                )

            # Create full date range
            start_date = pd.Timestamp("2009-03-01")
            end_date = pd.Timestamp.now()
            if interval == "day":
                full_range = pd.date_range(start=start_date, end=end_date, freq="D")
            else:
                full_range = pd.date_range(
                    start=start_date, end=end_date, freq=interval
                )

            # Reindex the dataframe with the full date range
            df = df.reindex(full_range)
            df.dropna(how="all", inplace=True)
            return df
        except Exception as e:
            logger.error(f"tt:get_data_for_x: an error occurred: {str(e)}")
            return pd.DataFrame()
