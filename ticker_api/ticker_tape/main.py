import calendar
import re
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, Any

import pytz
from dateutil.relativedelta import relativedelta, TH, WE, FR, MO
from redis import Redis
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from zerodha_api import ZerodhaConnect
from ticker_api.exceptions import InvalidExchangeException
from ticker_api.settings import (
    get_configuration,
)
from ticker_api.settings import get_logger
from ticker_api.ticker_database.schema import Instruments

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
    MAPPING_EQUITY_TO_FO_EXCHANGE = {"NSE": "NFO", "BSE": "BFO"}

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
        self.db_connection_string = f"mysql+pymysql://{db_username}:{db_password}@{db_host}/{self.db_schema_name}"
        self.engine = create_engine(
            self.db_connection_string,
            echo=False,
        )
        self.Session = sessionmaker(bind=self.engine)

        self.z_connect = ZerodhaConnect(token=token)
        redis_config = {
            "host": redis_host,
            "password": redis_password,
            "port": redis_port,
            "db": redis_db,
            "decode_responses": True,
        }
        self.redis_client = Redis(**redis_config)

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
        pattern = (
            r"^([A-Z]+)(?:\d{2}(?:[A-Z]{3}|[1-9OND])(?:\d{2,}(?:\.\d+)?[CP]E|FUT))$"
        )

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
                        spot_name=mapped_symbol,
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
                "tt:get_details: invalid params! please provide either (tradingsymbol, exchange) or instrument_token"
            )
