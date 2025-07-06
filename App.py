import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import websockets
import json
import requests
import time
from datetime import datetime, timedelta
import warnings
import hashlib
import hmac
import base64
import threading
from collections import deque
import queue
import ssl
import certifi
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from scipy.stats import norm
    import talib
except ImportError:
    st.error("Please install required packages: lightgbm, scikit-learn, scipy, TA-Lib")

# Set page config
st.set_page_config(
    page_title="Crypto Scalper Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-active { background-color: #00ff00; }
    .status-inactive { background-color: #ff0000; }
    .status-warning { background-color: #ffaa00; }
    .data-stream {
        background: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Bitget API Configuration
BITGET_API_KEY = "bg_9e976d2e57529796f0ad50d728d55a8c"
BITGET_SECRET_KEY = "e9eaff708fe070abf01f60b2f6afdb7ee917c62eef2172bbd70c34e8c726cad4"
BITGET_PASSPHRASE = "awaisaloloka"

# API Endpoints
BITGET_BASE_URL = "https://api.bitget.com"
BITGET_WS_URL = "wss://ws.bitget.com/spot/v1/stream"
MEMPOOL_API_URL = "https://mempool.space/api"
BITNODES_API_URL = "https://bitnodes.io/api"

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'selected_coins' not in st.session_state:
    st.session_state.selected_coins = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'live_data_store' not in st.session_state:
    st.session_state.live_data_store = None
if 'websocket_connections' not in st.session_state:
    st.session_state.websocket_connections = {}
if 'price_data' not in st.session_state:
    st.session_state.price_data = {}
if 'orderbook_data' not in st.session_state:
    st.session_state.orderbook_data = {}
if 'taker_flow_data' not in st.session_state:
    st.session_state.taker_flow_data = {}
if 'mempool_data' not in st.session_state:
    st.session_state.mempool_data = {}

# Bitget API Helper Functions
def generate_signature(timestamp, method, request_path, body, secret_key):
    if body is None:
        body = ""
    message = timestamp + method + request_path + body
    signature = base64.b64encode(hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()).decode()
    return signature

def get_bitget_headers(method, request_path, body=None):
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, request_path, body, BITGET_SECRET_KEY)
    
    headers = {
        'ACCESS-KEY': BITGET_API_KEY,
        'ACCESS-SIGN': signature,
        'ACCESS-TIMESTAMP': timestamp,
        'ACCESS-PASSPHRASE': BITGET_PASSPHRASE,
        'Content-Type': 'application/json'
    }
    return headers

# Live Data Store Class
class LiveDataStore:
    def __init__(self):
        self.price_data = {}
        self.orderbook_data = {}
        self.taker_flow_data = {}
        self.mempool_data = {}
        self.trade_history = {}
        self.market_depth = {}
        self.funding_rates = {}
        self.open_interest = {}
        self.liquidations = {}
        self.last_update = {}
        
    def update_price_data(self, symbol, price, volume, timestamp):
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=1000)
        
        self.price_data[symbol].append({
            'price': float(price),
            'volume': float(volume),
            'timestamp': timestamp
        })
        self.last_update[symbol] = timestamp
    
    def update_orderbook(self, symbol, bids, asks, timestamp):
        if not bids or not asks:
            return
            
        bid_prices = [float(bid[0]) for bid in bids[:5]]
        bid_volumes = [float(bid[1]) for bid in bids[:5]]
        ask_prices = [float(ask[0]) for ask in asks[:5]]
        ask_volumes = [float(ask[1]) for ask in asks[:5]]
        
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        
        self.orderbook_data[symbol] = {
            'bids': list(zip(bid_prices, bid_volumes)),
            'asks': list(zip(ask_prices, ask_volumes)),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': (best_ask - best_bid) / best_bid if best_bid > 0 else 0,
            'bid_volume': sum(bid_volumes),
            'ask_volume': sum(ask_volumes),
            'imbalance': (sum(bid_volumes) - sum(ask_volumes)) / (sum(bid_volumes) + sum(ask_volumes)) if sum(bid_volumes) + sum(ask_volumes) > 0 else 0,
            'timestamp': timestamp
        }
    
    def update_taker_flow(self, symbol, buy_volume, sell_volume, timestamp):
        if symbol not in self.taker_flow_data:
            self.taker_flow_data[symbol] = deque(maxlen=100)
        
        total_volume = buy_volume + sell_volume
        taker_buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        net_flow = buy_volume - sell_volume
        
        self.taker_flow_data[symbol].append({
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'taker_buy_ratio': taker_buy_ratio,
            'net_flow': net_flow,
            'timestamp': timestamp
        })
    
    def update_mempool_data(self, data):
        self.mempool_data = {
            'mempool_size': data.get('count', 0),
            'mempool_size_delta': data.get('vsize', 0) - self.mempool_data.get('vsize', 0),
            'fee_rate_fast': data.get('fastestFee', 0),
            'fee_rate_medium': data.get('halfHourFee', 0),
            'fee_rate_slow': data.get('hourFee', 0),
            'total_fee': data.get('totalFees', 0),
            'vsize': data.get('vsize', 0),
            'timestamp': time.time()
        }
    
    def get_recent_trades(self, symbol, count=50):
        if symbol not in self.price_data:
            return []
        return list(self.price_data[symbol])[-count:]
    
    def get_orderbook(self, symbol):
        return self.orderbook_data.get(symbol, {})
    
    def get_taker_flow(self, count=20):
        all_flows = []
        for symbol_flows in self.taker_flow_data.values():
            all_flows.extend(list(symbol_flows)[-count:])
        return all_flows
    
    def get_mempool_data(self):
        return self.mempool_data

# Initialize live data store
if st.session_state.live_data_store is None:
    st.session_state.live_data_store = LiveDataStore()

# Bitget WebSocket Client
class BitgetWebSocketClient:
    def __init__(self, symbols):
        self.symbols = symbols
        self.ws = None
        self.running = False
        self.data_store = st.session_state.live_data_store
        
    async def connect(self):
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.ws = await websockets.connect(
                BITGET_WS_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Subscribe to channels
            await self.subscribe_to_channels()
            self.running = True
            
            # Start listening
            await self.listen()
            
        except Exception as e:
            st.error(f"WebSocket connection error: {e}")
    
    async def subscribe_to_channels(self):
        subscription_messages = []
        
        for symbol in self.symbols:
            # Subscribe to ticker
            subscription_messages.append({
                "op": "subscribe",
                "args": [f"spot/ticker:{symbol}"]
            })
            
            # Subscribe to orderbook
            subscription_messages.append({
                "op": "subscribe", 
                "args": [f"spot/depth:{symbol}"]
            })
            
            # Subscribe to trades
            subscription_messages.append({
                "op": "subscribe",
                "args": [f"spot/trade:{symbol}"]
            })
        
        for msg in subscription_messages:
            await self.ws.send(json.dumps(msg))
    
    async def listen(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            st.warning("WebSocket connection closed")
        except Exception as e:
            st.error(f"WebSocket error: {e}")
    
    async def handle_message(self, data):
        try:
            if 'data' not in data:
                return
                
            channel = data.get('arg', '')
            message_data = data['data']
            
            if 'ticker' in channel:
                await self.handle_ticker(message_data)
            elif 'depth' in channel:
                await self.handle_orderbook(message_data)
            elif 'trade' in channel:
                await self.handle_trade(message_data)
                
        except Exception as e:
            st.error(f"Message handling error: {e}")
    
    async def handle_ticker(self, data):
        for item in data:
            symbol = item.get('instId')
            price = float(item.get('last', 0))
            volume = float(item.get('baseVol', 0))
            timestamp = time.time()
            
            self.data_store.update_price_data(symbol, price, volume, timestamp)
    
    async def handle_orderbook(self, data):
        for item in data:
            symbol = item.get('instId')
            bids = item.get('bids', [])
            asks = item.get('asks', [])
            timestamp = time.time()
            
            self.data_store.update_orderbook(symbol, bids, asks, timestamp)
    
    async def handle_trade(self, data):
        for item in data:
            symbol = item.get('instId')
            side = item.get('side')
            volume = float(item.get('sz', 0))
            timestamp = time.time()
            
            buy_volume = volume if side == 'buy' else 0
            sell_volume = volume if side == 'sell' else 0
            
            self.data_store.update_taker_flow(symbol, buy_volume, sell_volume, timestamp)

# API Data Fetchers
class DataFetcher:
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_bitget_symbols(self):
        try:
            endpoint = "/api/spot/v1/public/products"
            headers = get_bitget_headers("GET", endpoint)
            
            response = self.session.get(
                f"{BITGET_BASE_URL}{endpoint}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                symbols = [item['symbol'] for item in data.get('data', [])]
                return symbols
            else:
                st.error(f"Failed to fetch symbols: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error fetching symbols: {e}")
            return []
    
    def fetch_mempool_data(self):
        try:
            response = self.session.get(f"{MEMPOOL_API_URL}/mempool", timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.session_state.live_data_store.update_mempool_data(data)
                return data
            return {}
        except Exception as e:
            st.error(f"Error fetching mempool data: {e}")
            return {}
    
    def fetch_bitnodes_data(self):
        try:
            response = self.session.get(f"{BITNODES_API_URL}/v1/snapshots/latest/", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            st.error(f"Error fetching bitnodes data: {e}")
            return {}

# Initialize data fetcher
data_fetcher = DataFetcher()

# Main header
st.markdown('<div class="main-header">⚡ Ultra-Short-Term Crypto Scalper</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")

# Connection Status
st.sidebar.subheader("🔗 Connection Status")
if st.session_state.trading_active:
    st.sidebar.success("🟢 Live Data Connected")
    st.sidebar.info(f"📊 Streaming {len(st.session_state.selected_coins)} symbols")
else:
    st.sidebar.warning("🔴 Live Data Disconnected")

# Coin Selection
st.sidebar.subheader("🪙 Coin Selection")

# Fetch available coins from Bitget
@st.cache_data(ttl=3600)
def get_available_coins():
    symbols = data_fetcher.fetch_bitget_symbols()
    if symbols:
        return symbols[:30]  # Limit to 30 for performance
    else:
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT'
        ]

available_coins = get_available_coins()
selected_coins = st.sidebar.multiselect(
    "Select trading pairs:",
    available_coins,
    default=['BTCUSDT', 'ETHUSDT'],
    help="Select multiple coins for diversified scalping"
)

st.session_state.selected_coins = selected_coins

# Trading Parameters
st.sidebar.subheader("⚙️ Trading Parameters")
leverage = st.sidebar.slider("Leverage", min_value=1, max_value=200, value=10, step=1)
max_position_size = st.sidebar.slider("Max Position Size (%)", min_value=1, max_value=100, value=10, step=1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Time Parameters
st.sidebar.subheader("⏰ Time Parameters")
trade_duration = st.sidebar.selectbox("Trade Duration", ["30s", "1m", "2m", "5m"], index=2)
update_frequency = st.sidebar.selectbox("Update Frequency", ["30s", "1m"], index=0)

# Enhanced Feature Engineering with Live Data
class EnhancedFeatureEngineer:
    def __init__(self):
        self.kalman_filters = {}
        self.setar_models = {}
        self.addm_detectors = {}
        self.price_history = {}
        self.volume_history = {}
        
    def get_or_create_kalman_filter(self, symbol):
        if symbol not in self.kalman_filters:
            self.kalman_filters[symbol] = KalmanFilter()
        return self.kalman_filters[symbol]
    
    def get_or_create_setar_model(self, symbol):
        if symbol not in self.setar_models:
            self.setar_models[symbol] = SETARModel()
        return self.setar_models[symbol]
    
    def engineer_features(self, symbol):
        features = {}
        data_store = st.session_state.live_data_store
        
        # Price-based features
        recent_trades = data_store.get_recent_trades(symbol, 50)
        if recent_trades:
            prices = [trade['price'] for trade in recent_trades]
            volumes = [trade['volume'] for trade in recent_trades]
            
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                features['returns'] = returns[-1] if len(returns) > 0 else 0
                features['returns_lag1'] = returns[-2] if len(returns) > 1 else 0
                features['returns_lag2'] = returns[-3] if len(returns) > 2 else 0
                
                # Kalman filtered price
                kalman_filter = self.get_or_create_kalman_filter(symbol)
                filtered_price = kalman_filter.update(prices[-1])
                features['filtered_price'] = filtered_price
                features['price_deviation'] = (prices[-1] - filtered_price) / filtered_price
                
                # Volatility features
                if len(returns) >= 10:
                    features['volatility'] = np.std(returns[-10:])
                    features['volatility_ratio'] = np.std(returns[-5:]) / np.std(returns[-10:])
                else:
                    features['volatility'] = 0
                    features['volatility_ratio'] = 1
                
                # Volume features
                features['volume_mean'] = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
                features['volume_std'] = np.std(volumes[-10:]) if len(volumes) >= 10 else 0
                features['volume_ratio'] = volumes[-1] / np.mean(volumes[-10:]) if len(volumes) >= 10 and np.mean(volumes[-10:]) > 0 else 1
                
                # Regime detection
                setar_model = self.get_or_create_setar_model(symbol)
                regime = setar_model.detect_regime(returns)
                features['regime_momentum'] = 1 if regime == "momentum" else 0
                features['regime_mean_reversion'] = 1 if regime == "mean_reversion" else 0
                
                # Price momentum
                features['price_momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
                features['price_momentum_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        
        # Order book features
        orderbook = data_store.get_orderbook(symbol)
        if orderbook:
            features['bid_ask_spread'] = orderbook.get('spread', 0)
            features['order_imbalance'] = orderbook.get('imbalance', 0)
            features['bid_volume'] = orderbook.get('bid_volume', 0)
            features['ask_volume'] = orderbook.get('ask_volume', 0)
            features['best_bid'] = orderbook.get('best_bid', 0)
            features['best_ask'] = orderbook.get('best_ask', 0)
            
            # Market depth features
            if 'bids' in orderbook and 'asks' in orderbook:
                bid_depth = sum([vol for price, vol in orderbook['bids']])
                ask_depth = sum([vol for price, vol in orderbook['asks']])
                features['market_depth_ratio'] = bid_depth / ask_depth if ask_depth > 0 else 1
                features['weighted_mid_price'] = (orderbook['best_bid'] * ask_depth + orderbook['best_ask'] * bid_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        # Taker flow features
        taker_flows = data_store.get_taker_flow(20)
        if taker_flows:
            recent_ratios = [flow['taker_buy_ratio'] for flow in taker_flows[-10:]]
            recent_net_flows = [flow['net_flow'] for flow in taker_flows[-10:]]
            
            features['taker_buy_ratio'] = np.mean(recent_ratios) if recent_ratios else 0.5
            features['taker_buy_ratio_std'] = np.std(recent_ratios) if recent_ratios else 0
            features['taker_flow_trend'] = recent_ratios[-1] - recent_ratios[0] if len(recent_ratios) > 1 else 0
            features['net_taker_flow'] = np.mean(recent_net_flows) if recent_net_flows else 0
            features['net_taker_flow_momentum'] = recent_net_flows[-1] - recent_net_flo
