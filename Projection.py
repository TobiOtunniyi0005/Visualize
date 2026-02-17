import pygame
import numpy as np
from math import sqrt, sin, cos, radians
from screeninfo import get_monitors
from fractions import Fraction
import copy
import sys
import urllib.request
import json
import time
import pandas as pd
import os

# Initialize Pygame
try:
    pygame.init()
except Exception as e:
    print(f"Pygame initialization failed: {e}")
    sys.exit(1)

# Get primary monitor resolution
try:
    monitor = get_monitors()[0]
    monitor_width = monitor.width
    monitor_height = monitor.height
    print(f"Screen resolution: {monitor_width} x {monitor_height} pixels")
except Exception as e:
    print(f"Error getting monitor resolution: {e}")
    monitor_width, monitor_height = 1280, 720  # Fallback resolution

# API keys (replace with your own)
ALPHA_VANTAGE_API_KEY = "P3PD37S7548TZ4L0"  # Get from https://www.alphavantage.co
POLYGON_API_KEY = "O9gDA7yaLxy_9Zco6gOrGC0jJpKpgpOE"  # Get from https://polygon.io
TIINGO_API_KEY = "23f715daaa0c7b117ac8a29afbc184040ca547ff"  # Get from https://www.tiingo.com

# Top cryptocurrencies (CoinGecko ID, Yahoo Symbol, Binance Symbol)
top_cryptos = [
    ("bitcoin", "BTC-USD Bitcoin", "BTCUSDT"), ("ethereum", "ETH-USD Ethereum", "ETHUSDT"),
    ("ripple", "XRP-USD XRP", "XRPUSDT"), ("tether", "USDT-USD Tether USDt", "USDTUSD"),
    ("solana", "SOL-USD Solana", "SOLUSDT"), ("binancecoin", "BNB-USD BNB", "BNBUSDT"),
    ("usd-coin", "USDC-USD USD Coin", "USDCUSDT"), ("dogecoin", "DOGE-USD Dogecoin", "DOGEUSDT"),
    ("staked-ether", "STETH-USD Lido Staked ETH", "STETHUSDT"), ("cardano", "ADA-USD Cardano", "ADAUSDT"),
    ("tron", "TRX-USD TRON", "TRXUSDT"), ("wrapped-tron", "WTRX-USD Wrapped TRON", "WTRXUSDT"),
    ("hyperliquid", "HYPE32196-USD Hyperliquid", None), ("wsteth", "WSTETH-USD Lido wstETH", None),
    ("chainlink", "LINK-USD Chainlink", "LINKUSDT"), ("wrapped-beacon-eth", "WBETH-USD Wrapped Beacon ETH", None),
    ("weth", "WETH-USD WETH", "WETHUSDT"), ("wrapped-bitcoin", "WBTC-USD Wrapped Bitcoin", "WBTCUSDT"),
    ("sui", "SUI20947-USD Sui", "SUIUSDT"), ("usde", "USDE29470-USD Ethena USDe", None)
]
print("Top cryptocurrencies (CoinGecko ID, Yahoo Symbol, Binance Symbol):")
for cg_id, yahoo_name, binance_name in top_cryptos:
    print(f"{cg_id}: {yahoo_name}, {binance_name}")

# Font setup
font = pygame.font.Font(None, 28)
hint_font = pygame.font.Font(None, 24)
color_active = pygame.Color("dodgerblue2")
color_inactive = pygame.Color("lightskyblue3")

# Initial window size
init_width, init_height = min(1280, monitor_width), min(720, monitor_height)

# Input boxes for configuration
input_boxes = [
    {"rect": pygame.Rect(10, 10 + i * 60, 300, 32), "text": "", "active": i == 0, "label": label}
    for i, label in enumerate(["Symbols (e.g., BTC-USD,ETH-USD)", "Include time (y/n)", "Width,Height (e.g., 400,300)"])
]
symbols = None
include_time = False
userwidth = None
userheight = None
setting_config = True
try:
    screen = pygame.display.set_mode((init_width, init_height))
    pygame.display.set_caption("Configure Projection")
except Exception as e:
    print(f"Error setting display mode: {e}")
    sys.exit(1)
surface = None
scr_color = None
input_history = []
error_message = ""

# Camera settings
focal_length = 100.0
cam_pos = None
cam_dir = None
cam_yaw = 0.0
cam_pitch = 0.0
move_speed = 0.05
rotate_speed = 1.0
mouse_sensitivity = 0.05
points = None
point_colors = None
n_streams = None
x_dim = 0
y_dim = 1
z_dim = 2
mouse_captured = False

# Cache settings
CACHE_FILE = 'crypto_data.csv'
CACHE_TIMESTAMP_FILE = 'crypto_cache_timestamp.txt'
REQUEST_DELAY = 2  # Reduced for APIs with higher limits
RETRY_ATTEMPTS = 3

# Background color (light blue)
BACKGROUND_COLOR = (135, 206, 250)

def write_cache_timestamp():
    with open(CACHE_TIMESTAMP_FILE, 'w') as f:
        f.write(str(int(time.time())))

def read_cache_timestamp():
    if os.path.exists(CACHE_TIMESTAMP_FILE):
        with open(CACHE_TIMESTAMP_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def is_cache_fresh():
    timestamp = read_cache_timestamp()
    return (time.time() - timestamp) < 86400

def save_cache(df, timestamp):
    df.to_csv(CACHE_FILE)
    write_cache_timestamp()

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    return None

def fetch_historical_coingecko(cg_id, period1, period2):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart/range?vs_currency=usd&from={period1}&to={period2}&precision=4&interval=daily"
    try:
        print(f"Fetching {cg_id} from CoinGecko...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        prices = data.get('prices', [])
        if not prices:
            print(f"No price data returned for {cg_id}")
            return None
        timestamps = [p[0] // 1000 for p in prices]
        closes = [p[1] for p in prices]
        if not timestamps or not closes:
            print(f"Invalid data for {cg_id}: empty timestamps or closes")
            return None
        df = pd.DataFrame({'close': closes}, index=pd.to_datetime(np.array(timestamps), unit='s'))
        if df.empty:
            print(f"Empty DataFrame for {cg_id}")
            return None
        print(f"Fetched {cg_id}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {cg_id} from CoinGecko: {e}")
        return None

def fetch_historical_binance(symbol, period1, period2):
    symbol = symbol.replace('-USD', 'USDT')  # e.g., BTC-USD to BTCUSDT
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&startTime={period1*1000}&endTime={period2*1000}"
    try:
        print(f"Fetching {symbol} from Binance...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        timestamps = [pd.to_datetime(item[0], unit='ms') for item in data]
        closes = [float(item[4]) for item in data]
        df = pd.DataFrame({'close': closes}, index=timestamps)
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        print(f"Fetched {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from Binance: {e}")
        return None

def fetch_historical_alpha_vantage(symbol, period1, period2, api_key):
    url = f"https://www.alphavantage.co/query?function=CRYPTO_DAILY&symbol={symbol.replace('-USD', '')}&market=USD&apikey={api_key}"
    try:
        print(f"Fetching {symbol} from Alpha Vantage...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        timestamps = [pd.to_datetime(date) for date in data['Time Series (Digital Currency Daily)'].keys()]
        closes = [float(data['Time Series (Digital Currency Daily)'][date]['4. close']) for date in data['Time Series (Digital Currency Daily)'].keys()]
        df = pd.DataFrame({'close': closes}, index=timestamps)
        df = df.loc[(df.index >= pd.to_datetime(period1, unit='s')) & (df.index <= pd.to_datetime(period2, unit='s'))]
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        print(f"Fetched {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from Alpha Vantage: {e}")
        return None

def fetch_historical_tiingo(symbol, period1, period2, api_key):
    from_date = pd.to_datetime(period1, unit='s').strftime('%Y-%m-%d')
    to_date = pd.to_datetime(period2, unit='s').strftime('%Y-%m-%d')
    ticker = symbol.lower().replace('-usd', 'usd')
    url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={ticker}&startDate={from_date}&endDate={to_date}&resampleFreq=daily&token={api_key}"
    try:
        print(f"Fetching {symbol} from Tiingo...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        timestamps = [pd.to_datetime(item['date']) for item in data[0]['priceData']]
        closes = [item['close'] for item in data[0]['priceData']]
        df = pd.DataFrame({'close': closes}, index=timestamps)
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        print(f"Fetched {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from Tiingo: {e}")
        return None

def fetch_historical_polygon(symbol, period1, period2, api_key):
    from_date = pd.to_datetime(period1, unit='s').strftime('%Y-%m-%d')
    to_date = pd.to_datetime(period2, unit='s').strftime('%Y-%m-%d')
    ticker = f"X:{symbol.replace('-USD', 'USD')}"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={api_key}"
    try:
        print(f"Fetching {symbol} from Polygon...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        timestamps = [pd.to_datetime(item['t'], unit='ms') for item in data['results']]
        closes = [item['c'] for item in data['results']]
        df = pd.DataFrame({'close': closes}, index=timestamps)
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        print(f"Fetched {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from Polygon: {e}")
        return None

def fetch_historical_yahoo(symbol, period1, period2, interval='1d'):
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={period1}&period2={period2}&interval={interval}"
    try:
        print(f"Fetching {symbol} from Yahoo Finance...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        timestamps = data['chart']['result'][0]['timestamp']
        closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
        df = pd.DataFrame({'close': closes}, index=pd.to_datetime(np.array(timestamps), unit='s'))
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        print(f"Fetched {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} from Yahoo Finance: {e}")
        return None

def fetch_historical(symbol, period1, period2, interval='1d'):
    yahoo_to_cg = {yahoo.split()[0]: cg_id for cg_id, yahoo, _ in top_cryptos}
    yahoo_to_binance = {yahoo.split()[0]: binance for cg_id, yahoo, binance in top_cryptos}
    cg_id = yahoo_to_cg.get(symbol)
    binance_symbol = yahoo_to_binance.get(symbol)
    
    # Try APIs in order with exponential backoff
    for attempt in range(RETRY_ATTEMPTS):
        delay = 2 ** attempt * REQUEST_DELAY
        # CoinGecko
        if cg_id:
            df = fetch_historical_coingecko(cg_id, period1, period2)
            if df is not None:
                return df
        time.sleep(delay)
        # Binance
        if binance_symbol:
            df = fetch_historical_binance(binance_symbol, period1, period2)
            if df is not None:
                return df
        time.sleep(delay)
        # Alpha Vantage
        if ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_KEY":
            df = fetch_historical_alpha_vantage(symbol, period1, period2, ALPHA_VANTAGE_API_KEY)
            if df is not None:
                return df
        time.sleep(delay)
        # Tiingo
        if TIINGO_API_KEY != "YOUR_TIINGO_KEY":
            df = fetch_historical_tiingo(symbol, period1, period2, TIINGO_API_KEY)
            if df is not None:
                return df
        time.sleep(delay)
        # Polygon
        if POLYGON_API_KEY != "YOUR_POLYGON_KEY":
            df = fetch_historical_polygon(symbol, period1, period2, POLYGON_API_KEY)
            if df is not None:
                return df
        time.sleep(delay)
        # Yahoo Finance (last resort)
        df = fetch_historical_yahoo(symbol, period1, period2, interval)
        if df is not None:
            return df
        time.sleep(delay)
    print(f"All APIs failed for {symbol}")
    return None

def load_data(symbol_list, include_time, force_refresh=False):
    global points, point_colors, n_streams, cam_pos, cam_dir, error_message
    if not symbol_list:
        error_message = "Error: No symbols provided."
        print(error_message)
        return False
    
    now = int(time.time())
    period1 = now - 90 * 24 * 3600  # 3 months
    period2 = now
    interval = '1d'
    
    # Check cache
    if not force_refresh and is_cache_fresh():
        print("Using cached data (fresh within 1 day). Press 'r' to refresh.")
        full_df = load_cache()
        if full_df is not None and set(symbol_list).issubset(full_df.columns):
            main_sym = symbol_list[0]
            if main_sym not in full_df.columns:
                error_message = f"Main symbol {main_sym} not in cached data."
                print(error_message)
                return False
            ma_window = 20
            full_df['MA'] = full_df[main_sym].rolling(ma_window).mean().fillna(full_df[main_sym])
            print(f"Cached data loaded: {full_df.shape}, columns: {list(full_df.columns)}")
            
            colors = []
            for i in range(len(full_df)):
                rel = (full_df[main_sym].iloc[i] - full_df['MA'].iloc[i]) / full_df[main_sym].std()
                if rel > 0:
                    green = min(255, int(255 * min(1, rel * 5)))
                    colors.append([0, green, 0])
                else:
                    red = min(255, int(255 * min(1, -rel * 5)))
                    colors.append([red, 0, 0])
            point_colors = np.array(colors)
            
            if include_time:
                full_df['time'] = (full_df.index.astype(int) / 1e9 - period1) / (period2 - period1)
                cols = ['time'] + symbol_list + ['MA']
                norm_df = (full_df[cols] - full_df[cols].mean()) / full_df[cols].std().replace(0, 1)
            else:
                norm_df = (full_df[symbol_list] - full_df[symbol_list].mean()) / full_df[symbol_list].std().replace(0, 1)
            
            print(f"After processing: {norm_df.shape}, columns: {list(norm_df.columns)}")
            
            points = norm_df.values
            n_streams = points.shape[1]
            cam_pos = np.zeros(n_streams)
            cam_dir = np.zeros(n_streams)
            cam_dir[x_dim] = 1.0
            error_message = ""
            save_cache(full_df, now)
            return True
        print("Cache invalid or missing symbols. Fetching new data.")
    
    df_list = []
    for i, sym in enumerate(symbol_list):
        df = fetch_historical(sym, period1, period2, interval)
        if df is not None and not df.empty:
            df_list.append(df.rename(columns={'close': sym}))
        else:
            print(f"Failed to fetch data for {sym}")
        time.sleep(REQUEST_DELAY)
    
    if not df_list:
        error_message = "No valid data fetched for any symbols."
        print(error_message)
        return False
    
    try:
        full_df = pd.concat(df_list, axis=1, join='outer').ffill().dropna()
        if full_df.empty:
            error_message = "Concatenated DataFrame is empty after processing."
            print(error_message)
            return False
        print(f"Fetched data: {full_df.shape}, columns: {list(full_df.columns)}")
    except Exception as e:
        error_message = f"Error concatenating data: {e}"
        print(error_message)
        return False
    
    main_sym = symbol_list[0]
    if main_sym not in full_df.columns:
        error_message = f"Main symbol {main_sym} not in fetched data."
        print(error_message)
        return False
    
    ma_window = 20
    full_df['MA'] = full_df[main_sym].rolling(ma_window).mean().fillna(full_df[main_sym])
    print(f"After MA calculation: {full_df.shape}, columns: {list(full_df.columns)}")
    
    colors = []
    for i in range(len(full_df)):
        rel = (full_df[main_sym].iloc[i] - full_df['MA'].iloc[i]) / full_df[main_sym].std()
        if rel > 0:
            green = min(255, int(255 * min(1, rel * 5)))
            colors.append([0, green, 0])
        else:
            red = min(255, int(255 * min(1, -rel * 5)))
            colors.append([red, 0, 0])
    point_colors = np.array(colors)
    
    if include_time:
        full_df['time'] = (full_df.index.astype(int) / 1e9 - period1) / (period2 - period1)
        cols = ['time'] + symbol_list + ['MA']
        full_df = full_df[cols]
    
    print(f"After processing: {full_df.shape}, columns: {list(full_df.columns)}")
    
    means = full_df.mean()
    stds = full_df.std().replace(0, 1)
    norm_df = (full_df - means) / stds
    
    points = norm_df.values
    n_streams = points.shape[1]
    cam_pos = np.zeros(n_streams)
    cam_dir = np.zeros(n_streams)
    cam_dir[x_dim] = 1.0
    error_message = ""
    save_cache(full_df, now)
    print("Data saved to cache.")
    return True

def makearray(lister, default):
    templist = default
    for dim in reversed(lister):
        templist = [copy.deepcopy(templist) for _ in range(dim)]
    return templist

def update_camera_direction():
    global cam_dir
    yaw = radians(cam_yaw)
    pitch = radians(cam_pitch)
    cam_dir = np.zeros(n_streams)
    cam_dir[x_dim] = cos(pitch) * cos(yaw)
    cam_dir[z_dim] = cos(pitch) * sin(yaw)
    cam_dir[y_dim] = sin(pitch)
    norm = sqrt(sum(cam_dir**2))
    if norm > 0:
        cam_dir /= norm

def initialize_projection():
    global scr_color, surface
    if userwidth <= 0 or userheight <= 0:
        print(f"Error: Invalid dimensions {userwidth}x{userheight}")
        return
    try:
        scr_color = np.full((userheight, userwidth, 3), BACKGROUND_COLOR, dtype=np.uint8)
        print(f"scr_color shape: {scr_color.shape}, dtype: {scr_color.dtype}")
    except Exception as e:
        print(f"Error initializing scr_color: {e}")
        return
    
    surface = pygame.Surface((userwidth, userheight))
    surface.fill(BACKGROUND_COLOR)
    
    if points is not None and cam_dir is not None:
        rel_pos = points - cam_pos
        depth = np.dot(rel_pos, cam_dir)
        mask = (depth > 0.1) & (depth < 1000)
        if np.any(mask):
            factor = focal_length / depth[mask]
            factor = np.clip(factor, -1000, 1000)
            proj_x = (rel_pos[mask, x_dim] * factor + userwidth / 2.0).astype(int)
            proj_y = (rel_pos[mask, y_dim] * factor + userheight / 2.0).astype(int)
            valid_mask = (0 <= proj_x) & (proj_x < userwidth) & (0 <= proj_y) & (proj_y < userheight)
            if np.any(valid_mask):
                proj_x = proj_x[valid_mask]
                proj_y = proj_y[valid_mask]
                colors_valid = point_colors[mask][valid_mask]
                for px, py, color in zip(proj_x, proj_y, colors_valid):
                    pygame.draw.circle(surface, color, (px, py), 2)
    
    try:
        if surface is None:
            print("Error: Surface is None after initialization")
        else:
            print(f"Surface created: {surface.get_size()}")
    except Exception as e:
        print(f"Error validating surface: {e}")
        surface = None

def update_projection():
    update_camera_direction()
    initialize_projection()

running = True
clock = pygame.time.Clock()
last_refresh = 0
refresh_interval = 86400

while running:
    try:
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if setting_config:
                    for box in input_boxes:
                        box["active"] = box["rect"].collidepoint(event.pos)
                else:
                    if event.button == 3:  # Right-click
                        mouse_captured = not mouse_captured
                        pygame.mouse.set_visible(not mouse_captured)
                        if mouse_captured:
                            pygame.mouse.set_pos(userwidth // 2, userheight // 2)
            elif event.type == pygame.KEYDOWN:
                if setting_config:
                    active_box = next((box for box in input_boxes if box["active"]), None)
                    if active_box:
                        if event.key == pygame.K_RETURN:
                            try:
                                if active_box["label"] == "Symbols (e.g., BTC-USD,ETH-USD)":
                                    symbols = [s.strip() for s in active_box["text"].split(',')]
                                    if not symbols or not all(s for s in symbols):
                                        raise ValueError("No valid symbols provided.")
                                    valid_symbols = [yahoo.split()[0] for _, yahoo, _ in top_cryptos]
                                    invalid = [s for s in symbols if s not in valid_symbols]
                                    if invalid:
                                        raise ValueError(f"Invalid symbols: {', '.join(invalid)}")
                                    print(f"Symbols set: {symbols}")
                                    input_boxes[1]["active"] = True
                                    active_box["active"] = False
                                    active_box["text"] = ""
                                elif active_box["label"] == "Include time (y/n)":
                                    include_time = active_box["text"].strip().lower() == 'y'
                                    print(f"Include time set: {include_time}")
                                    input_boxes[2]["active"] = True
                                    active_box["active"] = False
                                    active_box["text"] = ""
                                elif active_box["label"] == "Width,Height (e.g., 400,300)":
                                    w_str, h_str = active_box["text"].split(",")
                                    userwidth = int(Fraction(w_str))
                                    userheight = int(Fraction(h_str))
                                    if userwidth <= 2 or userheight <= 2 or userwidth > monitor_width or userheight > monitor_height:
                                        raise ValueError("Invalid dimensions.")
                                    if not symbols:
                                        raise ValueError("Symbols not set. Please enter symbols first.")
                                    print(f"Dimensions set: {userwidth}x{userheight}")
                                    if load_data(symbols, include_time):
                                        try:
                                            screen = pygame.display.set_mode((userwidth, userheight))
                                            pygame.display.set_caption("Projection Visualization")
                                            initialize_projection()
                                            setting_config = False
                                            input_boxes = []
                                        except Exception as e:
                                            error_message = f"Error setting display mode: {e}"
                                            print(error_message)
                                            active_box["text"] = ""
                                    else:
                                        error_message = "Failed to load data. Check symbols or try again."
                                        print(error_message)
                                        active_box["text"] = ""
                            except (ValueError, ZeroDivisionError) as e:
                                error_message = f"Invalid input: {e}"
                                print(f"Invalid input for {active_box['label']}: {e}")
                        elif event.key == pygame.K_BACKSPACE:
                            active_box["text"] = active_box["text"][:-1]
                        else:
                            if active_box["label"] == "Symbols (e.g., BTC-USD,ETH-USD)":
                                if event.unicode.isalnum() or event.unicode in "-,":
                                    active_box["text"] += event.unicode
                            elif active_box["label"] == "Include time (y/n)":
                                if event.unicode in "ynYN":
                                    active_box["text"] += event.unicode
                            elif active_box["label"] == "Width,Height (e.g., 400,300)":
                                if event.unicode in "0123456789-,":
                                    active_box["text"] += event.unicode
                else:
                    if event.key == pygame.K_r and not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                        if load_data(symbols, include_time, force_refresh=True):
                            update_projection()
                    elif event.key == pygame.K_r and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                        cam_pos = np.zeros(n_streams)
                        cam_yaw = 0.0
                        cam_pitch = 0.0
                        focal_length = 100.0
                        update_projection()
                    elif event.key == pygame.K_x:
                        x_dim = (x_dim + 1) % n_streams
                        update_projection()
                    elif event.key == pygame.K_y:
                        y_dim = (y_dim + 1) % n_streams
                        update_projection()
                    elif event.key == pygame.K_z:
                        z_dim = (z_dim + 1) % n_streams
                        update_projection()

        if not setting_config:
            keys = pygame.key.get_pressed()
            speed_modifier = 0.5 if keys[pygame.K_LSHIFT] else 1.0
            update = False
            if keys[pygame.K_w]:
                cam_pos += cam_dir * move_speed * speed_modifier
                update = True
            if keys[pygame.K_s]:
                cam_pos -= cam_dir * move_speed * speed_modifier
                update = True
            if keys[pygame.K_a] or keys[pygame.K_d]:
                dir_local = np.array([cam_dir[x_dim], cam_dir[y_dim], cam_dir[z_dim]])
                up_local = np.array([0, 1, 0])
                cross_local = np.cross(dir_local, up_local)
                right = np.zeros(n_streams)
                right[x_dim] = cross_local[0]
                right[y_dim] = cross_local[1]
                right[z_dim] = cross_local[2]
                norm = np.sqrt(np.sum(right**2))
                if norm > 0:
                    right /= norm
                if keys[pygame.K_a]:
                    cam_pos -= right * move_speed * speed_modifier
                if keys[pygame.K_d]:
                    cam_pos += right * move_speed * speed_modifier
                update = True
            if keys[pygame.K_LEFT]:
                cam_yaw -= rotate_speed * speed_modifier
                update = True
            if keys[pygame.K_RIGHT]:
                cam_yaw += rotate_speed * speed_modifier
                update = True
            if keys[pygame.K_UP]:
                cam_pitch = min(89.0, cam_pitch + rotate_speed * speed_modifier)
                update = True
            if keys[pygame.K_DOWN]:
                cam_pitch = max(-89.0, cam_pitch - rotate_speed * speed_modifier)
                update = True
            if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
                focal_length += 10
                update = True
            if keys[pygame.K_MINUS]:
                focal_length = max(10, focal_length - 10)
                update = True
            if mouse_captured:
                mouse_dx, mouse_dy = pygame.mouse.get_rel()
                cam_yaw -= mouse_dx * mouse_sensitivity * speed_modifier
                cam_pitch = max(-89.0, min(89.0, cam_pitch - mouse_dy * mouse_sensitivity * speed_modifier))
                pygame.mouse.set_pos(userwidth // 2, userheight // 2)
                update = True
            if update:
                update_projection()

        if not setting_config and current_time - last_refresh > refresh_interval:
            if load_data(symbols, include_time):
                update_projection()
            last_refresh = current_time

        screen.fill(BACKGROUND_COLOR)
        print("Drawing screen...")
        if surface and not setting_config:
            screen.blit(surface, (0, 0))
            debug_texts = [
                f"Points: {len(points) if points is not None else 0}",
                f"Cam Pos: {cam_pos[x_dim]:.2f}, {cam_pos[y_dim]:.2f}, {cam_pos[z_dim]:.2f}",
                f"Cam Dir: {cam_dir[x_dim]:.2f}, {cam_dir[y_dim]:.2f}, {cam_dir[z_dim]:.2f}",
                f"Yaw: {cam_yaw:.1f}, Pitch: {cam_pitch:.1f}",
                f"Focal Length: {focal_length:.1f}",
                "Controls: WASD (move), Arrows (rotate), Shift (slow), r (refresh), Shift+R (reset), Right-click (mouse)"
            ]
            for i, text in enumerate(debug_texts):
                debug_surface = hint_font.render(text, True, (255, 255, 255))
                screen.blit(debug_surface, (10, 10 + i * 20))

        if setting_config:
            print("Rendering configuration UI...")
            for box in input_boxes:
                color = color_active if box["active"] else color_inactive
                pygame.draw.rect(screen, color, box["rect"], 2)
                display_text = box["text"]
                if font.size(display_text)[0] > box["rect"].w - 10:
                    display_text = display_text[:int(len(display_text) * (box["rect"].w - 10) / font.size(display_text)[0])] + "..."
                text_surface = font.render(f"{box['label']}: {display_text}", True, (255, 255, 255))
                screen.blit(text_surface, (box["rect"].x + 5, box["rect"].y + 5))
                box["rect"].w = min(init_width - 20, max(300, text_surface.get_width() + 10))
            
            hint_text = "Available symbols: " + ", ".join([yahoo.split()[0] for _, yahoo, _ in top_cryptos[:5]]) + "... (see console for full list)"
            hint_surface = hint_font.render(hint_text, True, (200, 200, 200))
            screen.blit(hint_surface, (10, init_height - 40))
            
            if error_message:
                error_surface = hint_font.render(error_message, True, (255, 100, 100))
                screen.blit(error_surface, (10, init_height - 70))

        pygame.display.flip()
        clock.tick(60)
    except Exception as e:
        print(f"Error in main loop: {e}")
        running = False

pygame.quit()