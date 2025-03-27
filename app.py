import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== PART 1: REVENUE PREDICTION ========================

def run_revenue_prediction(ticker):
    with st.spinner('Fetching revenue prediction data...'):
        # Fetch historical stock data
        data = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='3mo')
        
        # Feature Engineering
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data['Close_Lag3'] = data['Close'].shift(3)
        data['Price_Growth'] = data['Close'].pct_change()
        data['Revenue_Growth'] = data['Price_Growth']
        data.dropna(inplace=True)
        
        # Prepare features and target
        X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Price_Growth']]
        y = data['Close']
        
        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predict next quarter
        latest_data = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Price_Growth']].iloc[-1].values.reshape(1, -1)
        predicted_revenue = model.predict(latest_data)
        
        # Prepare results
        last_3_quarters = data.tail(3)
        result_df = last_3_quarters[['Close', 'Price_Growth', 'Revenue_Growth']]
        predicted_result = pd.DataFrame({'Close': [predicted_revenue[0]], 
                                        'Price_Growth': [None], 
                                        'Revenue_Growth': [None]}, 
                                      index=['Next Quarter'])
        result_df = pd.concat([result_df, predicted_result])
        
        return {
            'mae': mae,
            'predicted_revenue': predicted_revenue[0],
            'result_df': result_df,
            'full_data': data,
            'last_date': data.index[-1]
        }

# ======================== PART 2: EARNINGS SENTIMENT ========================

def fetch_earnings_dates(ticker):
    stock = yf.Ticker(ticker)
    earnings_dates = stock.earnings_dates
    if earnings_dates is None or earnings_dates.empty:
        raise ValueError(f"No earnings dates found for {ticker}")
    earnings_dates = earnings_dates.reset_index()[['Earnings Date']]
    earnings_dates['Earnings Date'] = pd.to_datetime(earnings_dates['Earnings Date'])
    earnings_dates = earnings_dates[earnings_dates['Earnings Date'] <= datetime.now(pytz.timezone('Asia/Kolkata'))]
    return earnings_dates.sort_values(by='Earnings Date', ascending=True)

def predict_next_earnings_date(earnings_dates):
    earnings_dates['interval'] = earnings_dates['Earnings Date'].diff().dt.days
    median_interval = earnings_dates['interval'].median()
    next_earnings_date = earnings_dates['Earnings Date'].iloc[-1] + timedelta(days=median_interval)
    return next_earnings_date

def fetch_stock_data(ticker):
    return yf.Ticker(ticker).history(period="200d")

def calculate_technical_indicators(data):
    # Moving Averages
    data['MA_50'] = ta.sma(data['Close'], length=50)
    data['MA_200'] = ta.sma(data['Close'], length=200)
    
    # RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    # Volume Z-Score
    data['Volume_Z'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    
    # MACD
    macd = ta.macd(data['Close'])
    data = pd.concat([data, macd], axis=1)
    
    return data.dropna()

def analyze_sentiment(data):
    sentiment_score = 0
    
    # Trend Analysis
    last_close = data['Close'].iloc[-1]
    if 'MA_50' in data and 'MA_200' in data:
        if last_close > data['MA_50'].iloc[-1] and last_close > data['MA_200'].iloc[-1]:
            sentiment_score += 2
        elif last_close > data['MA_50'].iloc[-1]:
            sentiment_score += 1
        elif last_close < data['MA_50'].iloc[-1] and last_close < data['MA_200'].iloc[-1]:
            sentiment_score -= 2
        else:
            sentiment_score -= 1
    
    # RSI Analysis
    if 'RSI' in data:
        rsi = data['RSI'].iloc[-1]
        if rsi > 70:
            sentiment_score -= 1
        elif rsi < 30:
            sentiment_score += 1
    
    # Volume Analysis
    if 'Volume_Z' in data and data['Volume_Z'].iloc[-1] > 2:
        sentiment_score += 1 if sentiment_score > 0 else -1
    
    # Sentiment Classification
    if sentiment_score >= 3:
        return "Very Good (Strong Bullish)"
    elif sentiment_score >= 1:
        return "Good (Bullish)"
    elif sentiment_score == 0:
        return "Neutral (Sideways)"
    elif sentiment_score >= -2:
        return "Bad (Bearish)"
    else:
        return "Very Bad (Strong Bearish)"

def run_earnings_sentiment(ticker):
    with st.spinner('Analyzing earnings sentiment...'):
        # Earnings Date Prediction
        try:
            earnings_dates = fetch_earnings_dates(ticker)
            next_earnings_date = predict_next_earnings_date(earnings_dates)
        except Exception as e:
            st.warning(f"{e}. Using default next earnings date.")
            next_earnings_date = datetime.now() + timedelta(days=90)
        
        # Stock Sentiment Analysis
        stock_data = fetch_stock_data(ticker)
        stock_data = calculate_technical_indicators(stock_data)
        sentiment = analyze_sentiment(stock_data)
        
        return {
            'next_earnings_date': next_earnings_date,
            'sentiment': sentiment,
            'stock_data': stock_data
        }

# ======================== STOCK SELECTION ========================

def get_stock_list():
    try:
        # Read the Excel file
        xls = pd.ExcelFile('stocklist.xlsx')
        
        # Get available sheets
        sheets = xls.sheet_names
        
        # Let user select a sheet
        selected_sheet = st.sidebar.selectbox("Select stock list", sheets)
        
        # Read the selected sheet
        df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
        
        # Get symbols
        symbols = df['Symbol'].tolist()
        
        # Let user select a stock
        selected_stock = st.sidebar.selectbox("Select stock to analyze", symbols)
        
        return selected_stock
    
    except FileNotFoundError:
        st.error("Error: 'stocklist.xlsx' file not found. Using default stock TCS.NS")
        return "TCS.NS"
    except Exception as e:
        st.error(f"Error reading stock list: {e}. Using default stock TCS.NS")
        return "TCS.NS"

# ======================== VISUALIZATION FUNCTIONS ========================

def plot_revenue_growth(data, last_date):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Revenue_Growth'] * 100, 
            label='Revenue Growth (%)', color='green', marker='o')
    ax.plot(data.index, data['Price_Growth'] * 100, 
            label='Price Growth (%)', color='blue', marker='x')
    ax.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Point')
    ax.set_title('Revenue Growth vs Price Growth')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth (%)')
    ax.legend()
    st.pyplot(fig)

def plot_revenue_prediction(data, last_date, predicted_revenue):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Close'], 
            label='Actual Revenue (Stock Price)', color='blue', marker='o')
    ax.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Point')
    next_date = last_date + pd.Timedelta(days=90)
    ax.scatter(next_date, predicted_revenue, 
              color='red', label='Predicted Revenue', zorder=5)
    ax.set_title('Stock Price (Proxy for Revenue) Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (Revenue Proxy)')
    ax.legend()
    st.pyplot(fig)

def plot_price_ma(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    if 'MA_50' in data:
        ax.plot(data.index, data['MA_50'], label='50-Day MA', color='orange', linestyle='--')
    if 'MA_200' in data:
        ax.plot(data.index, data['MA_200'], label='200-Day MA', color='red', linestyle='--')
    ax.set_title('Price and Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def plot_rsi(data):
    if 'RSI' in data:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['RSI'], label='RSI (14-day)', color='purple')
        ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        ax.set_title('Relative Strength Index (RSI)')
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI Value')
        ax.legend()
        st.pyplot(fig)

def plot_volume(data):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(data.index, data['Volume'], label='Volume', color='gray', alpha=0.7)
    if 'Volume_Z' in data:
        high_volume_days = data[data['Volume_Z'] > 2]
        if not high_volume_days.empty:
            ax.bar(high_volume_days.index, high_volume_days['Volume'], color='red', alpha=0.7, label='High Volume (Z>2)')
    ax.set_title('Trading Volume')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.legend()
    st.pyplot(fig)

# ======================== MAIN APP ========================

def main():
    st.title("📈 Comprehensive Stock Analysis Dashboard")
    
    # Get stock to analyze
    ticker = get_stock_list()
    
    st.header(f"Analysis for {ticker}")
    
    # Run both analyses
    revenue_results = run_revenue_prediction(ticker)
    sentiment_results = run_earnings_sentiment(ticker)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue Prediction")
        st.metric("Predicted Revenue for Next Quarter", f"{revenue_results['predicted_revenue']:,.2f}")
        st.metric("Model Mean Absolute Error", f"{revenue_results['mae']:.2f}")
        
        st.write("**Last 3 Quarters and Predicted Next Quarter:**")
        st.dataframe(revenue_results['result_df'].style.format({
            'Close': '{:,.2f}',
            'Price_Growth': '{:.2%}',
            'Revenue_Growth': '{:.2%}'
        }))
    
    with col2:
        st.subheader("📅 Earnings Sentiment")
        st.metric("Next Predicted Earnings Date", 
                 sentiment_results['next_earnings_date'].strftime('%Y-%m-%d'))
        
        sentiment_color = {
            "Very Good (Strong Bullish)": "green",
            "Good (Bullish)": "lightgreen",
            "Neutral (Sideways)": "gray",
            "Bad (Bearish)": "orange",
            "Very Bad (Strong Bearish)": "red"
        }
        st.metric("Current Sentiment", sentiment_results['sentiment'],
                 help="Based on technical indicators and price action")
        
        st.write("**Stock Technicals (Last Close):**")
        st.write(f"- Close Price: {sentiment_results['stock_data']['Close'].iloc[-1]:.2f}")
        if 'MA_50' in sentiment_results['stock_data']:
            st.write(f"- 50-Day MA: {sentiment_results['stock_data']['MA_50'].iloc[-1]:.2f}")
        if 'MA_200' in sentiment_results['stock_data']:
            st.write(f"- 200-Day MA: {sentiment_results['stock_data']['MA_200'].iloc[-1]:.2f}")
        if 'RSI' in sentiment_results['stock_data']:
            st.write(f"- RSI (14-day): {sentiment_results['stock_data']['RSI'].iloc[-1]:.2f}")
        if 'Volume_Z' in sentiment_results['stock_data']:
            st.write(f"- Volume Z-Score: {sentiment_results['stock_data']['Volume_Z'].iloc[-1]:.2f}")
    
    # Visualizations
    st.header("📈 Visualizations")
    
    st.subheader("Revenue Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_revenue_growth(revenue_results['full_data'], revenue_results['last_date'])
    with col2:
        plot_revenue_prediction(revenue_results['full_data'], revenue_results['last_date'], 
                              revenue_results['predicted_revenue'])
    
    st.subheader("Technical Analysis")
    plot_price_ma(sentiment_results['stock_data'])
    
    col1, col2 = st.columns(2)
    with col1:
        plot_rsi(sentiment_results['stock_data'])
    with col2:
        plot_volume(sentiment_results['stock_data'])

if __name__ == "__main__":
    main()
