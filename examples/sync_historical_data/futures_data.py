import argparse
import logging
from datetime import datetime
from pathlib import Path

from ticker_api import TickerDatabase
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
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Sync historical data with TickerDatabase for known symbols"
    )
    parser.add_argument("token", help="Zerodha API encrypted token")

    # Parse arguments
    args = parser.parse_args()

    # Setup external logger
    external_logger = setup_logger()

    # Configure the logger in ticker_api
    configure_logger(external_logger)

    # Initialize TickerDatabase with the provided token
    ticker_db = TickerDatabase(args.token)

    # List of symbols to sync
    names = [
        "NIFTY",
        "BANKNIFTY",
        "ABB",
        "ACC",
        "AUBANK",
        "AARTIIND",
        "ABBOTINDIA",
        "ADANIENT",
        "ADANIPORTS",
        "ABCAPITAL",
        "ABFRL",
        "ALKEM",
        "AMBUJACEM",
        "APOLLOHOSP",
        "APOLLOTYRE",
        "ASHOKLEY",
        "ASIANPAINT",
        "ASTRAL",
        "ATUL",
        "AUROPHARMA",
        "AXISBANK",
        "BSOFT",
        "BAJAJ-AUTO",
        "BAJFINANCE",
        "BAJAJFINSV",
        "BALKRISIND",
        "BALRAMCHIN",
        "BANDHANBNK",
        "BANKBARODA",
        "BATAINDIA",
        "BERGEPAINT",
        "BEL",
        "BHARATFORG",
        "BHEL",
        "BPCL",
        "BHARTIARTL",
        "BIOCON",
        "BOSCHLTD",
        "BRITANNIA",
        "CANFINHOME",
        "CANBK",
        "CHAMBLFERT",
        "CHOLAFIN",
        "CIPLA",
        "CUB",
        "COALINDIA",
        "COFORGE",
        "COLPAL",
        "CONCOR",
        "COROMANDEL",
        "CROMPTON",
        "CUMMINSIND",
        "DLF",
        "DABUR",
        "DALBHARAT",
        "DEEPAKNTR",
        "DIVISLAB",
        "DIXON",
        "LALPATHLAB",
        "DRREDDY",
        "EICHERMOT",
        "ESCORTS",
        "EXIDEIND",
        "GAIL",
        "GMRINFRA",
        "GLENMARK",
        "GODREJCP",
        "GODREJPROP",
        "GRANULES",
        "GRASIM",
        "GUJGASLTD",
        "GNFC",
        "HCLTECH",
        "HDFCAMC",
        "HDFCBANK",
        "HDFCLIFE",
        "HAVELLS",
        "HEROMOTOCO",
        "HINDALCO",
        "HAL",
        "HINDCOPPER",
        "HINDPETRO",
        "HINDUNILVR",
        "ICICIBANK",
        "ICICIGI",
        "ICICIPRULI",
        "IDFCFIRSTB",
        "IDFC",
        "IPCALAB",
        "ITC",
        "INDIAMART",
        "IEX",
        "IOC",
        "IRCTC",
        "IGL",
        "INDUSTOWER",
        "INDUSINDBK",
        "NAUKRI",
        "INFY",
        "INDIGO",
        "JKCEMENT",
        "JSWSTEEL",
        "JINDALSTEL",
        "JUBLFOOD",
        "KOTAKBANK",
        "LTF",
        "LTTS",
        "LICHSGFIN",
        "LTIM",
        "LT",
        "LAURUSLABS",
        "LUPIN",
        "MRF",
        "MGL",
        "M&MFIN",
        "M&M",
        "MANAPPURAM",
        "MARICO",
        "MARUTI",
        "MFSL",
        "METROPOLIS",
        "MPHASIS",
        "MCX",
        "MUTHOOTFIN",
        "NMDC",
        "NTPC",
        "NATIONALUM",
        "NAVINFLUOR",
        "NESTLEIND",
        "OBEROIRLTY",
        "ONGC",
        "OFSS",
        "PIIND",
        "PVRINOX",
        "PAGEIND",
        "PERSISTENT",
        "PETRONET",
        "PIDILITIND",
        "PEL",
        "POLYCAB",
        "PFC",
        "POWERGRID",
        "PNB",
        "RBLBANK",
        "RECLTD",
        "RELIANCE",
        "SBICARD",
        "SBILIFE",
        "SHREECEM",
        "SRF",
        "MOTHERSON",
        "SHRIRAMFIN",
        "SIEMENS",
        "SBIN",
        "SAIL",
        "SUNPHARMA",
        "SUNTV",
        "SYNGENE",
        "TATACONSUM",
        "TVSMOTOR",
        "TATACHEM",
        "TATACOMM",
        "TCS",
        "TATAMOTORS",
        "TATAPOWER",
        "TATASTEEL",
        "TECHM",
        "FEDERALBNK",
        "INDIACEM",
        "INDHOTEL",
        "RAMCOCEM",
        "TITAN",
        "TORNTPHARM",
        "TRENT",
        "UPL",
        "ULTRACEMCO",
        "UBL",
        "UNITDSPR",
        "VEDL",
        "IDEA",
        "VOLTAS",
        "WIPRO",
        "ZYDUSLIFE",
    ]

    # Expiry options
    expiry_options = ["CURRENT", "MID", "FAR"]

    # Loop through symbols and sync historical data
    for name_ in names:
        for expiry in expiry_options:
            params = {
                "exchange": "NFO",
                "segment": "NFO-FUT",
                "name": name_,
                "instrument_type": "FUT",
                "expiry": expiry,
                "interval": "day",
            }

            try:
                ticker_db.sync_historical_data(**params)
                print(
                    f"Successfully synced historical data for {name_} with expiry {expiry}"
                )
            except Exception as e:
                print(
                    f"Error syncing historical data for {name_} with expiry {expiry}: {str(e)}"
                )

    print("Historical data sync completed!")


if __name__ == "__main__":
    main()
