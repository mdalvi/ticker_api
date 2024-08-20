import logging

from ticker_api.settings import set_configuration
from ticker_api.ticker_database.main import TickerDatabase
from ticker_api.ticker_tape.main import TickerTape

# Create a null handler to avoid "No handler found" warnings.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

set_configuration()
