import logging
from datetime import datetime
from pathlib import Path

from ticker_api import TickerTape
from ticker_api import configure_logger


def setup_logger():
    # Create a logger
    logger = logging.getLogger("ticker_api")
    logger.setLevel(logging.INFO)

    # Create a file handler with datetime-based filename
    home_dir = Path.home()
    log_dir = home_dir / ".ticker_api" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{current_date}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatting for the logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def main():
    # Setup external logger
    external_logger = setup_logger()

    # Configure the logger in ticker_api
    configure_logger(external_logger)

    # Initialize TickerDatabase with the provided token
    ticker_tape = TickerTape()

    # Sync all existing historical data in database
    ticker_tape.get_data(tradingsymbol="ASIANPAINT", exchange="NSE")

    print("Get data operation completed!")


if __name__ == "__main__":
    main()
