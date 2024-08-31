import argparse
from ticker_api import TickerDatabase


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Sync historical FNO expiry date with TickerDatabase for known symbols")
    parser.add_argument("token", help="Zerodha API encrypted token")

    # Parse arguments
    args = parser.parse_args()

    # Initialize TickerDatabase with the provided token
    ticker_db = TickerDatabase(args.token)
    tradingsymbols = [
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

    for symbol in tradingsymbols:
        ticker_db.sync_historical_fno_expiry_dates(name=symbol)
    print('FNO dates sync success!')

if __name__ == "__main__":
    main()
