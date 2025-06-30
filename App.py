import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import tweepy
import praw
from pytrends.request import TrendReq
import yfinance as yf
from textblob import TextBlob

# -------------------------------
# 1. CONFIGURATION & API KEYS
# -------------------------------
BITGET_API_KEY      = st.secrets["BITGET_API_KEY"]
TWITTER_BEARER      = st.secrets["TWITTER_BEARER"]
REDDIT_CLIENT_ID    = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET= st.secrets["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT   = st.secrets["REDDIT_USER_AGENT"]
DUNE_API_KEY        = st.secrets["DUNE_API_KEY"]
SANTIMENT_API_KEY   = st.secrets["SANTIMENT_API_KEY"]
CMCAL_API_KEY       = st.secrets["CMCAL_API_KEY"]
CRYPTOPANIC_API_KEY = st.secrets["CRYPTOPANIC_API_KEY"]
LUNARCRUSH_API_KEY  = st.secrets["LUNARCRUSH_API_KEY"]
MESSARI_API_KEY     = st.secrets["MESSARI_API_KEY"]
THEGRAPH_ENDPOINT   = st.secrets["THEGRAPH_ENDPOINT"]

# -------------------------------
# 2. DATA FETCHING FUNCTIONS
# -------------------------------

def fetch_bitget_ohlcv(symbol: str, granularity: str, limit: int = 100):
    url = "https://api.bitget.com/api/spot/v1/market/candles"
    params = {"symbol": symbol, "granularity": granularity, "limit": limit}
    data = requests.get(url, params=params).json().get('data', [])
    df = pd.DataFrame(data, columns=['open_time','open','high','low','close','volume'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df.set_index('open_time')

def fetch_orderbook(symbol: str, limit: int = 50):
    url = "https://api.bitget.com/api/spot/v1/market/depth"
    params = {"symbol": symbol, "limit": limit}
    data = requests.get(url, params=params).json().get('data', {})
    bids = np.array(data.get('bids', []), dtype=float)
    asks = np.array(data.get('asks', []), dtype=float)
    return bids, asks

def fetch_defillama_tvl(chain: str = 'ethereum'):
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain.capitalize()}"
    resp = requests.get(url).json()
    df = pd.DataFrame(resp)
    if 'totalLiquidityUSD' in df.columns:
        df['date'] = pd.to_datetime(df['date'], unit='s')
        return df.set_index('date')['totalLiquidityUSD']
    return pd.Series()

def fetch_fear_and_greed_index():
    r = requests.get("https://api.alternative.me/fng/").json()
    return float(r['data'][0]['value'])

def fetch_dune_query(query_id: int):
    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    headers = {"Authorization": f"Bearer {DUNE_API_KEY}"}
    rows = requests.get(url, headers=headers).json().get('data', {}).get('rows', [])
    return float(rows[0][list(rows[0].keys())[0]]) if rows else 0

def fetch_santiment_metric(metric: str, days: int = 1):
    url = "https://api.santiment.net/graphql"
    query = {
        'query': f"""{{ getMetric(metric: "{metric}", from: "{(datetime.utcnow()-timedelta(days=days)).isoformat()}", 
            to: "{datetime.utcnow().isoformat()}", interval: "1d") {{ value }} }}"""
    }
    headers = {'Authorization': f'Apikey {SANTIMENT_API_KEY}'}
    data = requests.post(url, json=query, headers=headers).json()
    vals = [p['value'] for p in data.get('data', {}).get('getMetric', [])]
    return np.mean(vals) if vals else 0

def fetch_coinmarketcal_events():
    url = 'https://developers.coinmarketcal.com/v1/events'
    headers = {'x-api-key': CMCAL_API_KEY}
    body = requests.get(url, headers=headers).json().get('body', [])
    return len(body)

def fetch_cryptopanic_news(symbol: str):
    url = 'https://cryptopanic.com/api/v1/posts/'
    params = {'auth_token': CRYPTOPANIC_API_KEY, 'currencies': symbol.replace('USDT','')}
    results = requests.get(url, params=params).json().get('results', [])
    scores = [TextBlob(p.get('title','')).sentiment.polarity for p in results]
    return np.mean(scores) if scores else 0

def fetch_lunarcrush_asset(asset: str):
    url = "https://lunarcrush.com/api3/assets"
    params = {'symbol': asset.replace('USD',''), 'data':'market,social','key': LUNARCRUSH_API_KEY}
    data = requests.get(url, params=params).json().get('data', [])
    return data[0] if data else {}

def fetch_messari_metrics(asset='bitcoin'):
    url = f"https://data.messari.io/api/v1/assets/{asset}/metrics"
    headers = {"x-messari-api-key": MESSARI_API_KEY}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code==200:
            m = r.json().get('data',{}).get('metrics',{})
            md = m.get('market_data',{}); ch = m.get('blockchain_stats_24_hours',{})
            return {
                "price_usd": md.get("price_usd",0),
                "vol_usd": md.get("real_volume_last_24_hours",0),
                "tx_count": ch.get("transaction_count",0),
                "active_add": ch.get("active_addresses",0)
            }
    except: pass
    return {"price_usd":0,"vol_usd":0,"tx_count":0,"active_add":0}

def fetch_twitter_sentiment(query: str, start: str):
    client = tweepy.Client(bearer_token=TWITTER_BEARER)
    tweets = client.search_recent_tweets(query=query, start_time=start, max_results=50)
    texts = [t.text for t in tweets.data] if tweets.data else []
    scores = [TextBlob(t).sentiment.polarity for t in texts]
    return np.mean(scores) if scores else 0

def fetch_reddit_sentiment(subreddit: str):
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)
    posts = reddit.subreddit(subreddit).new(limit=50)
    texts = [p.title + (p.selftext or '') for p in posts]
    scores = [TextBlob(t).sentiment.polarity for t in texts]
    return np.mean(scores) if scores else 0

def fetch_google_trends(kws, tf='now 1-H'):
    pt = TrendReq()
    pt.build_payload(kws, timeframe=tf)
    df = pt.interest_over_time()
    return df[kws].iloc[-1].values if not df.empty else [0]*len(kws)

def fetch_macro_corr(symbol: str):
    pairs = [symbol.replace('USDT','-USD'),'SPY','QQQ','DX-Y.NYB']
    df = yf.download(pairs, period='1d', interval='5m')['Close'].pct_change().corr()
    return df.loc[pairs[0], pairs[1:]].values

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
TIMEFRAMES = {'5m':'300','10m':'600','15m':'900','1h':'3600','2h':'7200','4h':'14400','8h':'28800','1d':'86400'}

@st.cache_data
def make_feature_vector(sym: str, now: datetime):
    feats = []
    for label, gran in TIMEFRAMES.items():
        df = fetch_bitget_ohlcv(sym, gran, 20)
        if len(df)>1:
            ret = (df.close.iloc[-1]-df.close.iloc[-2])/df.close.iloc[-2]
            ma  = df.close.rolling(5).mean().iloc[-1]
        else:
            ret, ma = 0,0
        feats += [ret, ma]
    bids, asks = fetch_orderbook(sym)
    spread = asks[0,0]-bids[0,0] if bids.size else 0
    depth  = (bids[:5,1].sum()-asks[:5,1].sum())/(bids[:5,1].sum()+asks[:5,1].sum()) if bids.size else 0
    trades = pd.DataFrame(requests.get(f"https://api.bitget.com/api/spot/v1/market/fills?symbol={sym}&limit=100").json().get('data',[]))
    trades['size']=trades['size'].astype(float)
    trades['sign']=trades['side'].map({'buy':1,'sell':-1})
    oflow = (trades.sign*trades.size).sum() if not trades.empty else 0
    vpin  = abs(trades.sign.sum())/50 if not trades.empty else 0
    feats += [spread, depth, oflow, vpin]
    start = (now-timedelta(hours=1)).isoformat('T')+'Z'
    feats += [
        fetch_twitter_sentiment(sym, start),
        fetch_reddit_sentiment('CryptoCurrency'),
        fetch_fear_and_greed_index()
    ]
    gt_f, gt_c = fetch_google_trends(['finance','bitcoin'])
    feats += [gt_f, gt_c] + list(fetch_macro_corr(sym))
    # on-chain & events
    feats.append(fetch_dune_query(12345))
    feats.append(fetch_santiment_metric('market_earnings_usd'))
    feats.append(fetch_coinmarketcal_events())
    lunar = fetch_lunarcrush_asset(sym)
    feats += [lunar.get('galaxy_score',0), lunar.get('alt_rank',0)]
    feats.append(fetch_cryptopanic_news(sym))
    defillama = fetch_defillama_tvl(sym.replace('USDT','ethereum'))
    if len(defillama)>1:
        feats.append((defillama.iloc[-1]-defillama.iloc[-2])/defillama.iloc[-2])
    else:
        feats.append(0)
    # messari
    m = fetch_messari_metrics(sym.lower().replace('usdt',''))
    feats += [m['price_usd'], m['vol_usd'], m['tx_count'], m['active_add']]
    return np.array(feats)

# -------------------------------
# 4. MODEL & PREDICTION
# -------------------------------
@st.cache_resource
def get_model(ls, var):
    return GaussianProcessClassifier(kernel=C(var,(1e-3,1e3))*RBF(ls,(1e-3,1e3)))

# -------------------------------
# 5. STREAMLIT APP
# -------------------------------
def main():
    st.title("🚀 Multi-Coin Crypto Forecasting (GP)")
    coins = st.sidebar.multiselect("Coins", ['BTCUSDT','ETHUSDT','SOLUSDT'], default=['BTCUSDT'])
    ls = st.sidebar.slider("Length-Scale", 0.01, 10.0, 1.0)
    var= st.sidebar.slider("Variance", 0.1, 10.0, 1.0)
    hist=st.sidebar.slider("History Points", 50, 500, 100)
    horizon=st.sidebar.selectbox("Forecast Horizon", list(TIMEFRAMES.keys()), index=0)

    if st.sidebar.button("Run Forecast"):
        now = datetime.utcnow()
        scaler = StandardScaler()
        results = {}
        for sym in coins:
            X,y=[],[]
            for i in range(hist,0,-1):
                t = now - timedelta(minutes=i * (int(TIMEFRAMES[horizon])//(60 if 'h' in horizon else 1)))
                fv = make_feature_vector(sym, t)
                X.append(fv)
                dfn = fetch_bitget_ohlcv(sym, TIMEFRAMES[horizon], 2)
                y.append(int(float(dfn.close.iloc[-1])>float(dfn.close.iloc[-2])))
            X = np.vstack(X); y=np.array(y)
            Xs = scaler.fit_transform(X)
            model = get_model(ls,var)
            model.fit(Xs,y)
            fv_now = make_feature_vector(sym, now).reshape(1,-1)
            p = model.predict_proba(scaler.transform(fv_now))[0,1]
            results[sym] = p
        st.json({f"Up in {horizon}": results})

if __name__=="__main__":
    main()
