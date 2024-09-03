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
        "INDIA VIX",
        "NIFTY 50",
        "NIFTY BANK",
        "ABB INDIA",
        "ACC",
        "AU SMALL FINANCE BANK",
        "AARTI INDUSTRIES",
        "ABBOTT INDIA",
        "ADANI ENTERPRISES",
        "ADANI PORT & SEZ",
        "ADITYA BIRLA CAPITAL",
        "ADITYA BIRLA FASHION & RT",
        "ALKEM LABORATORIES",
        "AMBUJA CEMENTS",
        "APOLLO HOSPITALS ENTER. L",
        "APOLLO TYRES",
        "ASHOK LEYLAND",
        "ASIAN PAINTS",
        "ASTRAL",
        "ATUL",
        "AUROBINDO PHARMA",
        "AXIS BANK",
        "BIRLASOFT",
        "BAJAJ AUTO",
        "BAJAJ FINANCE",
        "BAJAJ FINSERV",
        "BALKRISHNA IND.",
        "BALRAMPUR CHINI MILLS",
        "BANDHAN BANK",
        "BANK OF BARODA",
        "BATA INDIA",
        "BERGER PAINTS (I)",
        "BHARAT ELECTRONICS",
        "BHARAT FORGE",
        "BHEL",
        "BHARAT PETROLEUM CORP  LT",
        "BHARTI AIRTEL",
        "BIOCON",
        "BOSCH",
        "BRITANNIA INDUSTRIES",
        "CAN FIN HOMES",
        "CANARA BANK",
        "CHAMBAL FERTILIZERS",
        "CHOLAMANDALAM IN & FIN CO",
        "CIPLA",
        "CITY UNION BANK",
        "COAL INDIA",
        "COFORGE",
        "COLGATE PALMOLIVE",
        "CONTAINER CORP OF IND",
        "COROMANDEL INTERNTL.",
        "CROMPT GREA CON ELEC",
        "CUMMINS INDIA",
        "DLF",
        "DABUR INDIA",
        "DALMIA BHARAT",
        "DEEPAK NITRITE",
        "DIVI S LABORATORIES",
        "DIXON TECHNO (INDIA)",
        "DR. LAL PATH LABS",
        "DR. REDDY S LABORATORIES",
        "EICHER MOTORS",
        "ESCORTS KUBOTA",
        "EXIDE INDUSTRIES",
        "GAIL (INDIA)",
        "GMR AIRPORTS INFRA",
        "GLENMARK PHARMACEUTICALS",
        "GODREJ CONSUMER PRODUCTS",
        "GODREJ PROPERTIES",
        "GRANULES INDIA",
        "GRASIM INDUSTRIES",
        "GUJARAT GAS",
        "GUJ NAR VAL FER & CHEM L",
        "HCL TECHNOLOGIES",
        "HDFC AMC",
        "HDFC BANK",
        "HDFC LIFE INS CO",
        "HAVELLS INDIA",
        "HERO MOTOCORP",
        "HINDALCO  INDUSTRIES ",
        "HINDUSTAN AERONAUTICS",
        "HINDUSTAN COPPER",
        "HINDUSTAN PETROLEUM CORP",
        "HINDUSTAN UNILEVER",
        "ICICI BANK",
        "ICICI LOMBARD GIC",
        "ICICI PRU LIFE INS CO",
        "IDFC FIRST BANK",
        "IDFC",
        "IPCA LABORATORIES",
        "ITC",
        "INDIAMART INTERMESH",
        "INDIAN ENERGY EXC",
        "INDIAN OIL CORP",
        "INDIAN RAIL TOUR CORP",
        "INDRAPRASTHA GAS",
        "INDUS TOWERS",
        "INDUSIND BANK",
        "INFO EDGE (I)",
        "INFOSYS",
        "INTERGLOBE AVIATION",
        "JK CEMENT",
        "JSW STEEL",
        "JINDAL STEEL & POWER",
        "JUBILANT FOODWORKS",
        "KOTAK MAHINDRA BANK",
        "L&T FINANCE",
        "L&T TECHNOLOGY SER.",
        "LIC HOUSING FINANCE",
        "LTIMINDTREE",
        "LARSEN & TOUBRO",
        "LAURUS LABS",
        "LUPIN",
        "MRF",
        "MAHANAGAR GAS",
        "M&M FIN. SERVICES",
        "MAHINDRA & MAHINDRA",
        "MANAPPURAM FINANCE",
        "MARICO",
        "MARUTI SUZUKI INDIA",
        "MAX FINANCIAL SERV",
        "METROPOLIS HEALTHCARE",
        "MPHASIS",
        "MULTI COMMODITY EXCHANGE",
        "MUTHOOT FINANCE",
        "NMDC",
        "NTPC",
        "NATIONAL ALUMINIUM CO",
        "NAVIN FLUORINE INT.",
        "NESTLE INDIA",
        "OBEROI REALTY",
        "OIL AND NATURAL GAS CORP.",
        "ORACLE FIN SERV SOFT",
        "PI INDUSTRIES",
        "PVR INOX",
        "PAGE INDUSTRIES",
        "PERSISTENT SYSTEMS",
        "PETRONET LNG",
        "PIDILITE INDUSTRIES",
        "PIRAMAL ENTERPRISES",
        "POLYCAB INDIA",
        "POWER FIN CORP",
        "POWER GRID CORP.",
        "PUNJAB NATIONAL BANK",
        "RBL BANK",
        "REC",
        "RELIANCE INDUSTRIES",
        "SBI CARDS & PAY SER",
        "SBI LIFE INSURANCE CO",
        "SHREE CEMENT",
        "SRF",
        "SAMVRDHNA MTHRSN INTL",
        "SHRIRAM FINANCE",
        "SIEMENS",
        "STATE BANK OF INDIA",
        "STEEL AUTHORITY OF INDIA",
        "SUN PHARMACEUTICAL IND L",
        "SUN TV NETWORK",
        "SYNGENE INTERNATIONAL",
        "TATA CONSUMER PRODUCT",
        "TVS MOTOR COMPANY ",
        "TATA CHEMICALS",
        "TATA COMMUNICATIONS",
        "TATA CONSULTANCY SERV LT",
        "TATA MOTORS",
        "TATA POWER CO",
        "TATA STEEL",
        "TECH MAHINDRA",
        "FEDERAL BANK",
        "THE INDIA CEMENTS",
        "THE INDIAN HOTELS CO.",
        "THE RAMCO CEMENTS",
        "TITAN COMPANY",
        "TORRENT PHARMACEUTICALS L",
        "TRENT",
        "UPL",
        "ULTRATECH CEMENT",
        "UNITED BREWERIES",
        "UNITED SPIRITS",
        "VEDANTA",
        "VODAFONE IDEA",
        "VOLTAS",
        "WIPRO",
        "ZYDUS LIFESCIENCES",
    ]

    # Loop through symbols and sync historical data
    for name_ in names:
        params = {
            "exchange": "NSE",
            "segment": (
                "INDICES" if name_ in ["INDIA VIX", "NIFTY 50", "NIFTY BANK"] else "NSE"
            ),
            "name": name_,
            "instrument_type": "EQ",
            "expiry": "NONE",
            "interval": "day",
        }

        try:
            ticker_db.sync_historical_data(**params)
            print(f"Successfully synced historical data for {name_}")
        except Exception as e:
            print(f"Error syncing historical data for {name_}: {str(e)}")

    print("Historical data sync completed!")


if __name__ == "__main__":
    main()
