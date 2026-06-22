# config/stocks_list.py
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO",
]

NIFTY_NEXT50 = [
    "ABB", "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BAJAJHLDNG",
    "BANKBARODA", "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "DABUR", "DLF", "DMART",
    "GAIL", "GODREJCP", "GODREJPROP", "HAL", "HAVELLS",
    "IRCTC", "JINDALSTEL", "LUPIN", "MARICO", "MUTHOOTFIN",
    "NAUKRI", "NHPC", "NMDC", "OFSS", "PAGEIND",
    "PERSISTENT", "PFC", "PIDILITIND", "PNB", "RECLTD",
    "SRF", "SBICARD", "SHREECEM", "SIEMENS", "TORNTPHARM",
    "TVSMOTOR", "VEDL", "VBL", "ZOMATO",
]

POPULAR_STOCKS = [
    "PAYTM", "POLICYBZR", "NYKAA", "MPHASIS", "COFORGE",
    "KPITTECH", "LTTS", "CYIENT", "BIOCON", "ALKEM",
    "IDFCFIRSTB", "FEDERALBNK", "RBLBANK", "MOTHERSON",
    "APOLLOTYRE", "MRF", "BALKRISIND", "IOC", "HINDPETRO",
    "GUJGASLTD", "IGL", "OBEROIRLTY", "PRESTIGE",
]

ALL_NSE_STOCKS = list(set(NIFTY50 + NIFTY_NEXT50 + POPULAR_STOCKS))

def get_yf_tickers(stocks=None, exchange="NS"):
    if stocks is None:
        stocks = ALL_NSE_STOCKS
    return [f"{s}.{exchange}" for s in stocks]

COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "INFY": "Infosys", "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "WIPRO": "Wipro", "BHARTIARTL": "Bharti Airtel", "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank", "HINDUNILVR": "Hindustan Unilever",
    "SBIN": "State Bank of India", "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki", "TATAMOTORS": "Tata Motors",
    "SUNPHARMA": "Sun Pharmaceutical", "DRREDDY": "Dr. Reddy's Laboratories",
    "NTPC": "NTPC Limited", "ONGC": "Oil & Natural Gas Corp",
    "POWERGRID": "Power Grid Corporation", "TECHM": "Tech Mahindra",
    "HCLTECH": "HCL Technologies", "TITAN": "Titan Company",
    "NESTLEIND": "Nestle India", "BAJAJ-AUTO": "Bajaj Auto",
    "HEROMOTOCO": "Hero MotoCorp", "EICHERMOT": "Eicher Motors",
    "M&M": "Mahindra & Mahindra", "TATASTEEL": "Tata Steel",
    "HINDALCO": "Hindalco Industries", "JSWSTEEL": "JSW Steel",
    "BPCL": "Bharat Petroleum", "COALINDIA": "Coal India",
    "CIPLA": "Cipla", "DIVISLAB": "Divi's Laboratories",
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports",
    "ASIANPAINT": "Asian Paints", "BRITANNIA": "Britannia Industries",
    "LTIM": "LTIMindtree", "ULTRACEMCO": "UltraTech Cement",
    "ZOMATO": "Zomato", "PAYTM": "Paytm (One97 Communications)",
    "NAUKRI": "Info Edge (Naukri)", "DMART": "Avenue Supermarts",
    "IRCTC": "IRCTC", "HAL": "Hindustan Aeronautics",
}

def get_company_name(ticker: str) -> str:
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    return COMPANY_NAMES.get(symbol, symbol)
