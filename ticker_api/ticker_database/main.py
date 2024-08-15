from redis import Redis
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from zerodha_api import ZerodhaConnect

from ticker_api.settings import (
    get_configuration,
)
from ticker_api.settings import get_logger
from ticker_api.ticker_database.schema import Base

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
