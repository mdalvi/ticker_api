from contextlib import contextmanager
from decimal import Decimal
from typing import Optional, Dict, Any, List

from redis import Redis
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from zerodha_api import ZerodhaConnect

from ticker_api.exceptions import MultipleTokensFoundException
from ticker_api.settings import (
    get_configuration,
)
from ticker_api.settings import get_logger
from ticker_api.ticker_database.schema import Instruments

logger = get_logger()


class TickerTape:
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

    def get_details(
        self,
        tradingsymbol: Optional[str] = None,
        exchange: Optional[str] = None,
        instrument_token: Optional[int] = None,
        return_db_details: bool = False,
    ) -> Dict[str, Any]:
        if tradingsymbol is not None and exchange is not None:
            return self._get_details_by_symbol(
                tradingsymbol, exchange, return_db_details
            )
        elif instrument_token is not None:
            return self._get_details_by_token(instrument_token, return_db_details)
        else:
            raise ValueError(
                "tt:get_details: invalid params! please provide either (tradingsymbol, exchange) or instrument_token"
            )

    def _get_details_by_symbol(
        self, tradingsymbol: str, exchange: str, return_db_details: bool
    ) -> Dict[str, Any]:
        try:
            with self._session_scope() as session:
                # Construct the query to get the instrument_tokens
                query = select(Instruments.instrument_token).where(
                    (Instruments.tradingsymbol == tradingsymbol)
                    & (Instruments.exchange == exchange)
                )

                # Execute the query
                results: List[int] = session.execute(query).scalars().all()

                if not results:
                    return {}  # Return an empty dict if no matching instrument is found
                elif len(results) > 1:
                    raise MultipleTokensFoundException(
                        f"tt:_get_details_by_symbol: multiple tokens found for tradingsymbol '{tradingsymbol}' and exchange '{exchange}': {results}"
                    )

                # Call _get_details_by_token with the found instrument_token
                return self._get_details_by_token(results[0], return_db_details)

        except SQLAlchemyError as e:
            logger.info(
                f"tt:_get_details_by_symbol: an error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception
        except MultipleTokensFoundException as e:
            logger.info(str(e))
            raise  # Re-raise the exception after printing the error message

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

                exclude_keys = {"id", "status", "updated_at", "created_at"}
                result_dict = {
                    column.name: getattr(result, column.name)
                    for column in Instruments.__table__.columns
                    if return_db_details or column.name not in exclude_keys
                }
                for key, value in result_dict.items():
                    if isinstance(value, Decimal):
                        result_dict[key] = float(value)
                return result_dict

        except SQLAlchemyError as e:
            logger.info(
                f"tt:_get_details_by_token: An error occurred while fetching instrument details: {str(e)}"
            )
            return {}  # Return an empty dict in case of any exception
