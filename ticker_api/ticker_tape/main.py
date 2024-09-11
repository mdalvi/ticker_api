import math
import re
from contextlib import contextmanager
from datetime import datetime, date, time
from decimal import Decimal
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pytz
from sqlalchemy import select, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from ticker_api.exceptions import (
    InvalidExchangeException,
)
from ticker_api.settings import get_configuration, get_logger
from ticker_api.ticker_database.schema import (
    Instruments,
    HistoricalData,
    FNOExpiryDates,
)

logger = get_logger()


class TickerTape:
    SUPPORTED_EXCHANGES = {"NSE", "NFO"}
    MAPPING_INDICES_TO_FNO_NAMES = {
        "NIFTY 50": "NIFTY",
        "NIFTY BANK": "BANKNIFTY",
    }
    MAPPING_EQUITY_TO_FNO_EXCHANGE = {"NSE": "NFO"}
    MAPPING_EQUITY_TO_FNO_SEGMENT = {"NSE": "NFO-FUT"}

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

    def _get_monthly_expiry_date(
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
                    f"tt::get_monthly_expiry_date:: returning `None` since not expiry length specified in the request."
                )
                return None

            attr_str = f"{exchange}|{segment}|{name}|FUT|{expiry}"
            with self._session_scope() as session:
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
                        f"tt::get_monthly_expiry_date:: no expiry dates found for the given parameters: {attr_str}"
                    )
                    return None

                expiry_dates = [row[0] for row in result]

                if len(expiry_dates) < 3:
                    logger.warning(
                        f"tt::get_monthly_expiry_date:: less than three expiry dates found for the given parameters: {attr_str}"
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
                        f"tt::get_monthly_expiry_date:: invalid expiry type: {expiry}. Must be CURRENT, MID, or FAR."
                    )
                    return None

        except SQLAlchemyError as e:
            logger.error(
                f"tt::get_monthly_expiry_date:: error in get_expiry_date: {str(e)}"
            )
            return None

    def _get_next_three_active_contracts(
        self,
        exchange: str,
        fno_name: str,
        spot_name: str,
    ) -> list:
        """
        Generate the names of the next three active futures contracts.
        """
        logger.info(
            f"tt::_get_next_three_active_contracts:: fetching next three contract names for {exchange}|{fno_name}|{spot_name}"
        )
        # Get relative date (current date) in the specified timezone
        relative_date = datetime.now(pytz.timezone(self.config["timezone"])).date()
        fno_exchange = self.MAPPING_EQUITY_TO_FNO_EXCHANGE.get(exchange, exchange)
        segment = self.MAPPING_EQUITY_TO_FNO_SEGMENT.get(exchange, exchange)

        # Generate the next three active contract names
        contract_names = []
        for expiry_str in ["CURRENT", "MID", "FAR"]:
            contract_date = self._get_monthly_expiry_date(
                exchange=fno_exchange,
                segment=segment,
                name=fno_name,
                expiry=expiry_str,
                relative_date=relative_date,
            )
            if contract_date:
                contract_name = (
                    f'{spot_name}{contract_date.strftime("%y%b")}FUT'.upper()
                )
                contract_names.append(contract_name)
        logger.info(
            f"tt::_get_next_three_active_contracts:: returning next three contracts viz. {contract_names}"
        )

        return contract_names

    def _get_details_by_symbol(
        self,
        tradingsymbol: str,
        exchange: str,
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
                    if column.name not in exclude_keys
                }
                for key, value in result_dict.items():
                    if isinstance(value, Decimal):
                        result_dict[key] = float(value)

                if fetch_futures:
                    fno_name = self.MAPPING_INDICES_TO_FNO_NAMES.get(
                        tradingsymbol, tradingsymbol
                    )  # For futures contracts the tradingsymbol is the name of the underlying of the contract.
                    futures_contract_names = self._get_next_three_active_contracts(
                        exchange=exchange,
                        fno_name=fno_name,
                        spot_name=self._extract_spot_name(fno_name),
                    )

                    futures_contracts = []
                    for future_symbol in futures_contract_names:
                        fno_exchange = self.MAPPING_EQUITY_TO_FNO_EXCHANGE.get(
                            exchange, exchange
                        )
                        future_details = self.get_details(future_symbol, fno_exchange)
                        if future_details:
                            futures_contracts.append(future_details)
                    result_dict["futures_contracts"] = futures_contracts

                return result_dict

        except SQLAlchemyError as e:
            logger.info(
                f"tt::_get_details_by_symbol: An error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception

    def _get_details_by_token(self, instrument_token: int) -> Dict[str, Any]:
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
                        f"tt::_get_details_by_token: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                    )

                # Determine whether to fetch futures based on the tradingsymbol
                fetch_futures = not tradingsymbol.endswith("FUT")

                # Use _get_details_by_symbol to get all details including futures
                return self._get_details_by_symbol(
                    tradingsymbol, exchange, fetch_futures
                )
        except SQLAlchemyError as e:
            logger.info(
                f"tt::_get_details_by_token: An error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception

    def get_details(
        self,
        tradingsymbol: Optional[str] = None,
        exchange: Optional[str] = None,
        instrument_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get details of an instrument from the database based on either (tradingsymbol and exchange) or (instrument token).

        :param tradingsymbol: The trading symbol of the instrument.
        :param exchange: The exchange where the instrument is traded.
        :param instrument_token: The unique token identifying the instrument.
        :return: A dictionary containing the instrument details.
        """
        if tradingsymbol is not None and exchange is not None:
            if exchange not in self.SUPPORTED_EXCHANGES:
                raise InvalidExchangeException(
                    f"tt::get_details: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                )
            fetch_futures = not tradingsymbol.endswith("FUT")
            return self._get_details_by_symbol(tradingsymbol, exchange, fetch_futures)
        elif instrument_token is not None:
            return self._get_details_by_token(instrument_token)
        else:
            raise ValueError(
                "tt::get_details: invalid parameters! please provide either (tradingsymbol, exchange) or instrument_token"
            )

    def _get_dte(
        self, exchange: str, segment: str, name: str, relative_date: date, timezone: str
    ) -> float:
        """
        Calculate the number of days to expiry for the current active contract.
        """
        # Get current date and time in the specified timezone
        tz = pytz.timezone(timezone)
        record_datetime = tz.localize(datetime.combine(relative_date, time(15, 15, 0)))

        # Get the expiry date of the current month
        relative_exp_date = self._get_monthly_expiry_date(
            exchange, segment, name, expiry="CURRENT", relative_date=relative_date
        )
        # Create a datetime object for the expiry at 15:30:00
        relative_exp_datetime = tz.localize(
            datetime.combine(relative_exp_date, time(15, 30, 0))
        )

        # Calculate the time difference
        time_difference = relative_exp_datetime - record_datetime

        # Convert the time difference to days (float)
        dte = time_difference.total_seconds() / (24 * 3600)

        return dte

    @staticmethod
    def _get_historical_data(
        session: Session,
        exchange: str,
        segment: str,
        name: str,
        instrument_type: str,
        expiry: str,
        interval: str,
    ) -> pd.DataFrame:

        attr_str = f"{exchange}|{segment}|{name}|{instrument_type}|{expiry}|{interval}"
        query = select(
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
            (HistoricalData.exchange == exchange)
            & (HistoricalData.segment == segment)
            & (HistoricalData.name == name)
            & (HistoricalData.instrument_type == instrument_type)
            & (HistoricalData.expiry == expiry)
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
            logger.warning(f"tt::_get_historical_data: no data found for {attr_str}")
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
            logger.warning("tt::_get_vix_data:: india vix instrument not found")
            return pd.DataFrame()

        logger.info(
            f"tt::_get_vix_data:: querying for vix data of INDIA VIX@NSE with interval {interval}"
        )
        df = self._get_historical_data(
            session,
            exchange=vix_instrument.exchange,
            segment=vix_instrument.segment,
            name=vix_instrument.name,
            instrument_type=vix_instrument.instrument_type,
            expiry="NONE",
            interval=interval,
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
                "tt::_get_vix_data:: duplicate index values found. keeping last occurrence."
            )
            df = df[~df.index.duplicated(keep="last")]

        return df

    def _get_futures_data(
        self,
        session: Session,
        instrument_details: Dict[str, Any],
        interval: str,
    ) -> pd.DataFrame:

        tradingsymbol = instrument_details["tradingsymbol"]
        exchange = instrument_details["exchange"]
        futures_contracts: List[Dict[str, Any]] = instrument_details[
            "futures_contracts"
        ]
        expiry_notation = ["CURRENT", "MID", "FAR"]

        dfs = []
        for i, futures_contract in enumerate(futures_contracts):

            fut_tradingsymbol = futures_contract["tradingsymbol"]
            fut_exchange = futures_contract["exchange"]
            fut_segment = futures_contract["segment"]
            fut_instrument_name = futures_contract["name"]
            fut_instrument_type = futures_contract["instrument_type"]

            logger.info(
                f"tt::_get_futures_data:: querying for futures data of {fut_tradingsymbol}@{fut_exchange} with interval {interval}"
            )
            df = self._get_historical_data(
                session,
                exchange=fut_exchange,
                segment=fut_segment,
                name=fut_instrument_name,
                instrument_type=fut_instrument_type,
                expiry=expiry_notation[i],
                interval=interval,
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
                f"tt::_get_futures_data:: no futures data found for {tradingsymbol} on {exchange}"
            )
            return pd.DataFrame()

        combined_df = pd.concat(dfs, axis=1)

        # Ensure index is unique
        if combined_df.index.duplicated().any():
            logger.warning(
                "tt::_get_futures_data:: duplicate index values found. keeping last occurrence."
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

        if futures_contracts:
            renamed_df["fut_dte"] = renamed_df.index.map(
                lambda date_: self._get_dte(
                    exchange=futures_contracts[0]["exchange"],
                    segment=futures_contracts[0]["segment"],
                    name=futures_contracts[0]["name"],
                    relative_date=date_,
                    timezone=self.config["timezone"],
                )
            )
        else:
            renamed_df["fut_dte"] = np.nan

        return renamed_df

    def _get_spot_data(
        self, session: Session, instrument_details: Dict[str, Any], interval: str
    ) -> pd.DataFrame:

        tradingsymbol = instrument_details["tradingsymbol"]
        exchange = instrument_details["exchange"]
        segment = instrument_details["segment"]
        instrument_name = instrument_details["name"]
        instrument_type = instrument_details["instrument_type"]

        logger.info(
            f"tt::_get_spot_data:: querying for equity data of {tradingsymbol}@{exchange} with interval {interval}"
        )

        df = self._get_historical_data(
            session,
            exchange=exchange,
            segment=segment,
            name=instrument_name,
            instrument_type=instrument_type,
            expiry="NONE",
            interval=interval,
        )

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
                "tt::_get_spot_data:: duplicate index values found. keeping last occurrence."
            )
            df = df[~df.index.duplicated(keep="last")]

        return df

    @staticmethod
    def _adjust_data_for_splits(df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the ratio between futures and spot prices
        df["fut_price_ratio"] = df["fut_close"] / df["spot_close"]

        def to_nearest_x1_multiple(value: float) -> float:
            # Check if the value is np.nan or pd.NA
            if pd.isna(value):
                return np.nan

            # Get the order of magnitude of the value
            order = math.floor(math.log10(abs(value)))

            # Calculate the x1 multiple for this order of magnitude
            x1_multiple = 10**order

            # Round to the nearest x1 multiple
            return round(value / x1_multiple) * x1_multiple

        # Apply the split ratio estimation
        df["fut_estimated_split"] = df["fut_price_ratio"].apply(to_nearest_x1_multiple)

        # Adjust the futures price
        df["fut_open"] = df["fut_open"] / df["fut_estimated_split"]
        df["fut_high"] = df["fut_high"] / df["fut_estimated_split"]
        df["fut_low"] = df["fut_low"] / df["fut_estimated_split"]
        df["fut_close"] = df["fut_close"] / df["fut_estimated_split"]
        df["fut_close_mid"] = df["fut_close_mid"] / df["fut_estimated_split"]
        df["fut_close_far"] = df["fut_close_far"] / df["fut_estimated_split"]
        df["fut_volume"] = df["fut_volume"] * df["fut_estimated_split"]
        df["fut_oi"] = df["fut_oi"] * df["fut_estimated_split"]
        df["spot_volume"] = df["spot_volume"] * df["fut_estimated_split"]
        return df

    def _get_data_by_symbol(
        self, tradingsymbol: str, exchange: str, interval: str
    ) -> pd.DataFrame:
        try:
            equity_details = self.get_details(tradingsymbol, exchange)
            index_details = self.get_details("NIFTY 50", "NSE")
            if equity_details:
                with self._session_scope() as session:

                    spot_data_df = self._get_spot_data(
                        session, equity_details, interval
                    )

                    index_spot_df = self._get_spot_data(
                        session, index_details, interval
                    )
                    if index_spot_df.columns:
                        index_spot_df.columns = index_spot_df.columns.str.replace(
                            "^spot_", "index_spot_", regex=True
                        )

                    fut_data_df = self._get_futures_data(
                        session,
                        equity_details,
                        interval,
                    )
                    index_fut_data_df = self._get_futures_data(
                        session,
                        index_details,
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
            else:
                logger.warning(
                    f"tt::_get_data_by_symbol:: equity instrument not found for {tradingsymbol} on {exchange}"
                )
        except SQLAlchemyError as e:
            logger.error(
                f"tt::_get_data_for_x_by_symbol:: an error occurred in sqlalchemy operations: {str(e)}"
            )
            return pd.DataFrame()

    def _get_data_by_token(self, instrument_token: int, interval: str) -> pd.DataFrame:
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
                        f"tt::_get_data_for_x_by_token:: invalid exchange {exchange}, supported types are {self.SUPPORTED_EXCHANGES}"
                    )

                # Use _get_data_for_x_by_symbol to get all data
                return self._get_data_by_symbol(tradingsymbol, exchange, interval)
        except SQLAlchemyError as e:
            logger.info(
                f"tt::_get_data_for_x_by_token:: an error occurred while fetching data for x: {str(e)}"
            )
            return (
                pd.DataFrame()
            )  # Return an empty pd.DataFrame in case of any exception

    @staticmethod
    def _standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        dtype_mapping = {"Float64": "float64", "Int64": "int64", "UInt32": "uint32"}

        for column in df.columns:
            current_dtype = str(df[column].dtype)
            if current_dtype in dtype_mapping:
                try:
                    df[column] = df[column].astype(dtype_mapping[current_dtype])
                except ValueError as e:
                    logger.warning(
                        f"tt::_standardize_dtypes:: could not convert column {column} to {dtype_mapping[current_dtype]}. Error: {e}"
                    )
                    # If conversion fails, we might want to handle NaN values
                    if pd.api.types.is_float_dtype(df[column]):
                        df[column] = df[column].astype("float64")
                    elif pd.api.types.is_integer_dtype(df[column]):
                        # For integer columns with NaN, we need to use float
                        df[column] = df[column].astype("float64")

        # Convert any remaining object columns to appropriate types
        for column in df.select_dtypes(include=["object"]):
            try:
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                # If conversion to numeric fails, leave as object
                pass

        return df

    def get_data(
        self,
        tradingsymbol: Optional[str] = None,
        exchange: Optional[str] = None,
        instrument_token: Optional[int] = None,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Returns combined `OHLCV + OI` pd.DataFrame of equity, index, futures and vix data with date index

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
                        f"tt::get_data:: Invalid exchange {exchange}"
                    )
                df = self._get_data_by_symbol(tradingsymbol, exchange, interval)
            elif instrument_token:
                df = self._get_data_by_token(instrument_token, interval)
            else:
                raise ValueError(
                    "tt::get_data:: invalid parameters! provide either (tradingsymbol, exchange) or instrument_token"
                )

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            df["fut_oi"] = df["fut_oi"].replace(0.0, np.nan).ffill().fillna(0)
            df["index_oi"] = df["index_oi"].replace(0.0, np.nan).ffill().fillna(0)
            df["spot_volume"] = df["spot_volume"].replace(0.0, np.nan).ffill().fillna(0)
            df["fut_volume"] = df["fut_volume"].replace(0.0, np.nan).ffill().fillna(0)
            df["index_volume"] = (
                df["index_volume"].replace(0.0, np.nan).ffill().fillna(0)
            )
            df = df.drop(["index_spot_volume"], axis=1)

            df = self._standardize_dtypes(df)
            return df
        except Exception as e:
            logger.error(f"tt::get_data:: an error occurred: {str(e)}")
            return pd.DataFrame()
