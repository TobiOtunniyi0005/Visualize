import yfinance as yf
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Streamlit app setup
st.set_page_config(page_title="Stock Tracker with High-Dim Insights", layout="wide")
st.title("Stock Price Tracker with High-Dim Insights")
st.markdown("Track stocks in real-time and discover trading insights using high-dimensional analysis.")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,TSLA,AMZN")
alert_price = st.sidebar.number_input("Set alert price for first stock", value=150.0, step=1.0)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:5]  # Limit to 5 tickers

# Manual indicator calculations
def calculate_ma20(data):
    return data['Close'].rolling(window=20).mean()

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    return 100 - (100 / (1 + rs))

# Cache data fetching for performance
@st.cache_data
def fetch_data(tickers, period="30d", interval="1d"):
    data_dict = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                st.warning(f"No data for {ticker}. Skipping.")
                continue
            hist['MA20'] = calculate_ma20(hist)
            hist['RSI'] = calculate_rsi(hist)
            features = hist[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI']].dropna()
            if features.empty:
                st.warning(f"No valid features for {ticker} after dropping NaNs.")
                continue
            data_dict[ticker] = features.mean()
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    df = pd.DataFrame(data_dict).T
    if df.empty:
        st.error("All tickers failed to fetch valid data. Using mock data for demo.")
        mock_data = {
            'AAPL': [150.0, 152.0, 149.0, 151.0, 1000000, 150.5, 60.0],
            'MSFT': [300.0, 302.0, 298.0, 301.0, 800000, 300.5, 55.0],
            'GOOGL': [2500.0, 2520.0, 2480.0, 2510.0, 500000, 2505.0, 58.0]
        }
        df = pd.DataFrame(mock_data, index=['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI']).T
    return df

# High-dimensional visualization
if tickers:
    st.header("High-Dimensional Stock Clustering")
    high_dim_data = fetch_data(tickers)
    if not high_dim_data.empty:
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(high_dim_data)
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'], index=high_dim_data.index)
            reduced_df['Daily Return'] = (high_dim_data['Close'] - high_dim_data['Open']) / high_dim_data['Open'] * 100
            fig = px.scatter(reduced_df, x='PC1', y='PC2', color='Daily Return',
                            hover_name=reduced_df.index, title="Stock Clustering (PCA)",
                            color_continuous_scale='RdYlGn', size_max=10)
            fig.update_traces(marker=dict(size=15), textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Insight**: Stocks cluster by metrics like price and RSI. Outliers signal trading opportunities (e.g., high volatility). Hover to explore.")
        except Exception as e:
            st.error(f"Error in PCA visualization: {e}")
    else:
        st.error("No valid data for high-dim visualization. Check ticker inputs or try different tickers.")

# Single-stock tracker
if tickers:
    st.header(f"Price Tracker for {tickers[0]}")
    try:
        stock = yf.Ticker(tickers[0])
        data = stock.history(period="5d", interval="1h")
        if not data.empty:
            latest_price = data["Close"].iloc[-1]
            fig_price = px.line(data, x=data.index, y="Close", title=f"{tickers[0]} Stock Price (Last 5 Days)")
            st.plotly_chart(fig_price, use_container_width=True)
            if latest_price < alert_price:
                st.error(f"Alert! {tickers[0]} price dropped to {latest_price:.2f}")
            else:
                st.success(f"Current price: {latest_price:.2f}")
        else:
            st.warning(f"No data available for {tickers[0]}. Try a different ticker.")
    except Exception as e:
        st.error(f"Error fetching data for {tickers[0]}: {e}")