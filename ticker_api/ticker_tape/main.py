import calendar
import re
from contextlib import contextmanager
from datetime import datetime, date, time
from decimal import Decimal
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta, TH, WE, FR, MO
from sqlalchemy import select, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from ticker_api.exceptions import (
    InvalidExchangeException,
    InvalidInstrumentCategoryException,
)
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
        """
        # Pattern for futures and options
        pattern = r"^((?:[A-Z]+\s?)+)(?:\d{2}(?:[A-Z]{3}|[1-9OND])(?:\d{2,}(?:\.\d+)?[CP]E|FUT))?$"
        match = re.match(pattern, tradingsymbol)
        if match:
            return match.group(1)

        return tradingsymbol  # If no pattern matches, return the original symbol

    @contextmanager
    def _session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        """
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
    def _get_last_trading_day(year, month, expiry_weekday):
        """
        Function to get the last trading day of a given month
        """
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        return last_day + relativedelta(weekday=expiry_weekday(-1))

    def _get_next_three_active_contracts(
            self, spot_name: str, expiry_day: str, timezone: str
    ) -> list:
        """
        Generate the names of the next three active futures contracts.
        """

        # Set up the expiry day mapping
        expiry_day_map = {"Monday": MO, "Wednesday": WE, "Thursday": TH, "Friday": FR}
        expiry_weekday = expiry_day_map.get(
            expiry_day, TH
        )  # Default to Thursday if invalid day provided

        # Get current date in the specified timezone
        current_date = datetime.now(pytz.timezone(timezone)).date()

        # Get the expiry date of the current month
        current_expiry = self._get_last_trading_day(
            current_date.year, current_date.month, expiry_weekday
        )

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

    def _get_dte(self, record_date: date, expiry_day: str, timezone: str) -> float:
        """
        Calculate the number of days to expiry for the current active contract.
        """
        # Set up the expiry day mapping
        expiry_day_map = {"Monday": MO, "Wednesday": WE, "Thursday": TH, "Friday": FR}
        expiry_weekday = expiry_day_map.get(
            expiry_day, TH
        )  # Default to Thursday if invalid day provided

        # Get current date and time in the specified timezone
        tz = pytz.timezone(timezone)
        record_datetime = tz.localize(datetime.combine(record_date, time(15, 15, 0)))

        # Get the expiry date of the current month
        record_expiry = self._get_last_trading_day(
            record_datetime.year, record_datetime.month, expiry_weekday
        )

        # If current date is past the current month's expiry, get next month's expiry
        if record_datetime.date() > record_expiry:
            if record_datetime.month == 12:
                next_expiry = self._get_last_trading_day(
                    record_datetime.year + 1, 1, expiry_weekday
                )
            else:
                next_expiry = self._get_last_trading_day(
                    record_datetime.year, record_datetime.month + 1, expiry_weekday
                )
        else:
            next_expiry = record_expiry

        # Create a datetime object for the expiry at 15:30:00
        expiry_datetime = tz.localize(datetime.combine(next_expiry, time(15, 30, 0)))

        # Calculate the time difference
        time_difference = expiry_datetime - record_datetime

        # Convert the time difference to days (float)
        days_to_expiry = time_difference.total_seconds() / (24 * 3600)

        return days_to_expiry

    def _get_details_by_symbol(
            self,
            tradingsymbol: str,
            exchange: str,
            return_db_details: bool,
            fetch_futures: bool,
    ) -> Dict[str, Any]:
        """
        Retrieves instrument details from the database based on the trading symbol and exchange.

        This method queries the database for an instrument matching the given trading symbol
        and exchange. It can optionally include additional details and fetch related futures
        contracts.
        """
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
        """
        Retrieves instrument details from the database based on the instrument token.

        This method queries the database for an instrument matching the given instrument token,
        then uses the retrieved trading symbol and exchange to fetch full details.
        """
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
        Get details of an instrument from the database based on either (tradingsymbol and exchange) or (instrument token).

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
    def _check_supported_category(session: Session, tradingsymbol: str, exchange: str):
        instrument = session.execute(
            select(Instruments).where(
                (Instruments.tradingsymbol == tradingsymbol)
                & (Instruments.exchange == exchange)
            )
        ).scalar_one_or_none()

        if not instrument:
            raise ValueError(
                f"tt:_check_supported_category: instrument not found for {tradingsymbol} on {exchange}"
            )

        segment = instrument.segment
        instrument_type = instrument.instrument_type
        if not (exchange == "NSE" and segment == "NSE" and instrument_type == "EQ"):
            raise InvalidInstrumentCategoryException(
                f"tt:_check_supported_category: invalid category: {exchange}|{segment}|{instrument_type}"
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

    def _get_futures_data(
            self,
            session: Session,
            tradingsymbol: str,
            exchange: str,
            interval: str,
    ) -> pd.DataFrame:
        mapped_symbol, expiry_day = self.MAPPING_INDICES_TO_FO_SYMBOLS.get(
            (tradingsymbol, exchange), (tradingsymbol, "Thursday")
        )
        future_symbols = self._get_next_three_active_contracts(
            spot_name=self._extract_spot_name(mapped_symbol),
            expiry_day=expiry_day,
            timezone=self.config["timezone"],
        )

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
                elif i == 1:
                    df = df[["close_price", "oi"]].rename(
                        columns={"oi": "oi_mid", "close_price": "close_price_mid"}
                    )
                elif i == 2:
                    df = df[["close_price"]].rename(
                        columns={"close_price": "close_price_far"}
                    )
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
            "close_price_mid": "fut_close_mid",
            "close_price_far": "fut_close_far",
        }
        renamed_df = combined_df.rename(columns=fut_columns)[list(fut_columns.values())]
        renamed_df["fut_dte"] = renamed_df.index.map(
            lambda date_: self._get_dte(date_, expiry_day, self.config["timezone"])
        )
        return renamed_df

    def _get_spot_data(
            self, session: Session, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        logger.info(
            f"tt:_get_spot_data: querying for equity data of {tradingsymbol}@{exchange} with interval {interval}"
        )
        instrument = session.execute(
            select(Instruments).where(
                (Instruments.tradingsymbol == tradingsymbol)
                & (Instruments.exchange == exchange)
            )
        ).scalar_one_or_none()

        if not instrument:
            logger.warning(
                f"tt:_get_spot_data: equity instrument not found for {tradingsymbol} on {exchange}"
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
                "tt:_get_spot_data: duplicate index values found. keeping last occurrence."
            )
            df = df[~df.index.duplicated(keep="last")]

        return df

    @staticmethod
    def _adjust_data_for_splits(df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the ratio between futures and spot prices
        df["fut_price_ratio"] = df["fut_close"] / df["spot_close"]

        def estimate_split_ratio(ratio: float) -> float:
            if ratio > 1.5:
                # Consider split ratios from 2 to 100
                possible_splits = np.arange(2, 101)
                # Find the split ratio that minimizes the difference
                best_split = min(possible_splits, key=lambda x: abs(ratio - x))
                return best_split * 1.0
            return 1.0

        # Apply the split ratio estimation
        df["fut_estimated_split"] = df["fut_price_ratio"].apply(estimate_split_ratio)

        # Adjust the futures price
        df["fut_open"] = df["fut_open"] / df["fut_estimated_split"]
        df["fut_high"] = df["fut_high"] / df["fut_estimated_split"]
        df["fut_low"] = df["fut_low"] / df["fut_estimated_split"]
        df["fut_close"] = df["fut_close"] / df["fut_estimated_split"]
        df["fut_volume"] = df["fut_volume"] * df["fut_estimated_split"]
        df["fut_oi"] = df["fut_oi"] * df["fut_estimated_split"]
        df["spot_volume"] = df["spot_volume"] * df["fut_estimated_split"]

        return df

    def _get_data_for_x_by_symbol(
            self, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        try:
            with self._session_scope() as session:
                self._check_supported_category(session, tradingsymbol, exchange)

                spot_data_df = self._get_spot_data(
                    session, tradingsymbol, exchange, interval
                )
                index_spot_df = self._get_spot_data(
                    session, "NIFTY 50", "NSE", interval
                )
                index_spot_df.columns = index_spot_df.columns.str.replace(
                    "^spot_", "index_spot_", regex=True
                )
                fut_data_df = self._get_futures_data(
                    session,
                    tradingsymbol,
                    exchange,
                    interval,
                )
                index_fut_data_df = self._get_futures_data(
                    session,
                    "NIFTY 50",
                    "NSE",
                    interval,
                )
                index_fut_data_df.columns = index_fut_data_df.columns.str.replace(
                    "^fut_", "index_", regex=True
                )
                vix_data_df = self._get_vix_data(session, interval)
                complete_df = pd.concat(
                    [
                        spot_data_df,
                        index_spot_df,
                        fut_data_df,
                        index_fut_data_df,
                        vix_data_df,
                    ],
                    axis=1,
                    verify_integrity=True,
                )
                complete_df = self._adjust_data_for_splits(complete_df)
                return complete_df

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
            return df
        except Exception as e:
            logger.error(f"tt:get_data_for_x: an error occurred: {str(e)}")
            return pd.DataFrame()
