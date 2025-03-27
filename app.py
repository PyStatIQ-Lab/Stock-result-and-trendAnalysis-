import streamlit as st
import yfinance as yf
import pandas as pd
import ta  # Alternative to pandas_ta that works better with Streamlit
from datetime import datetime, timedelta
import pytz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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
        data = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), interval='3mo')
        
        # Feature Engineering
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data['Close_Lag3'] = data['Close'].shift(3)
        data['Price_Growth'] = data['Close'].pct_change()
        data['Revenue_Growth'] = data['Price_Growth']
        data.dropna(inplace=True)
        
        X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Price_Growth']]
        y = data['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        latest_data = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Price_Growth']].iloc[-1].values.reshape(1, -1)
        predicted_revenue = model.predict(latest_data)
        
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

def calculate_technical_indicators(data):
    # Initialize indicators
    indicator = ta.TrendIndicator()
    
    # Moving Averages
    data['MA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['MA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # Volume Z-Score
    data['Volume_Z'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    
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
        return "Very Good (Strong Bullish)", "🟢"
    elif sentiment_score >= 1:
        return "Good (Bullish)", "🟡"
    elif sentiment_score == 0:
        return "Neutral (Sideways)", "⚪"
    elif sentiment_score >= -2:
        return "Bad (Bearish)", "🟠"
    else:
        return "Very Bad (Strong Bearish)", "🔴"

def run_earnings_sentiment(ticker):
    with st.spinner('Analyzing earnings sentiment...'):
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="200d")
        stock_data = calculate_technical_indicators(stock_data)
        sentiment, emoji = analyze_sentiment(stock_data)
        
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                earnings = earnings.reset_index()[['Earnings Date']]
                earnings['Earnings Date'] = pd.to_datetime(earnings['Earnings Date'])
                earnings = earnings[earnings['Earnings Date'] <= datetime.now(pytz.timezone('Asia/Kolkata'))]
                earnings['interval'] = earnings['Earnings Date'].diff().dt.days
                median_interval = earnings['interval'].median()
                next_earnings_date = earnings['Earnings Date'].iloc[-1] + timedelta(days=median_interval)
            else:
                next_earnings_date = datetime.now() + timedelta(days=90)
        except:
            next_earnings_date = datetime.now() + timedelta(days=90)
        
        return {
            'next_earnings_date': next_earnings_date,
            'sentiment': sentiment,
            'sentiment_emoji': emoji,
            'stock_data': stock_data
        }

# ======================== STOCK SELECTION ========================

def get_stock_list():
    try:
        xls = pd.ExcelFile('stocklist.xlsx')
        sheets = xls.sheet_names
        selected_sheet = st.sidebar.selectbox("Select stock list", sheets)
        df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
        symbols = df['Symbol'].tolist()
        selected_stock = st.sidebar.selectbox("Select stock to analyze", symbols)
        return selected_stock
    except:
        return "TCS.NS"

# ======================== VISUALIZATION ========================

def create_visualizations(revenue_results, sentiment_results):
    # Revenue Plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue Growth vs Price Growth")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(revenue_results['full_data'].index, 
                revenue_results['full_data']['Revenue_Growth']*100, 
                label='Revenue Growth (%)', color='green')
        ax.plot(revenue_results['full_data'].index, 
                revenue_results['full_data']['Price_Growth']*100, 
                label='Price Growth (%)', color='blue')
        ax.axvline(x=revenue_results['last_date'], color='gray', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue Prediction")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(revenue_results['full_data'].index, 
                revenue_results['full_data']['Close'], 
                label='Actual Revenue')
        next_date = revenue_results['last_date'] + pd.Timedelta(days=90)
        ax.scatter(next_date, revenue_results['predicted_revenue'], 
                  color='red', label='Predicted')
        ax.legend()
        st.pyplot(fig)
    
    # Technical Plots
    st.subheader("Technical Indicators")
    
    # Price and MAs
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(sentiment_results['stock_data'].index, 
            sentiment_results['stock_data']['Close'], 
            label='Price')
    if 'MA_50' in sentiment_results['stock_data']:
        ax.plot(sentiment_results['stock_data'].index, 
                sentiment_results['stock_data']['MA_50'], 
                label='50 MA')
    if 'MA_200' in sentiment_results['stock_data']:
        ax.plot(sentiment_results['stock_data'].index, 
                sentiment_results['stock_data']['MA_200'], 
                label='200 MA')
    ax.legend()
    st.pyplot(fig)
    
    # RSI and Volume
    col1, col2 = st.columns(2)
    with col1:
        if 'RSI' in sentiment_results['stock_data']:
            st.subheader("RSI (14-day)")
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(sentiment_results['stock_data'].index, 
                    sentiment_results['stock_data']['RSI'], 
                    color='purple')
            ax.axhline(70, color='red', linestyle='--')
            ax.axhline(30, color='green', linestyle='--')
            st.pyplot(fig)
    
    with col2:
        st.subheader("Volume")
        fig, ax = plt.subplots(figsize=(10,3))
        ax.bar(sentiment_results['stock_data'].index, 
               sentiment_results['stock_data']['Volume'], 
               color='gray')
        if 'Volume_Z' in sentiment_results['stock_data']:
            high_vol = sentiment_results['stock_data']['Volume_Z'] > 2
            ax.bar(sentiment_results['stock_data'].index[high_vol], 
                   sentiment_results['stock_data']['Volume'][high_vol], 
                   color='red')
        st.pyplot(fig)

# ======================== MAIN APP ========================

def main():
    st.title("📈 Stock Analysis Dashboard")
    ticker = get_stock_list()
    
    st.header(f"Analysis for {ticker}")
    
    revenue_results = run_revenue_prediction(ticker)
    sentiment_results = run_earnings_sentiment(ticker)
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Next Quarter", f"${revenue_results['predicted_revenue']:,.2f}")
    with col2:
        st.metric("Next Earnings Date", sentiment_results['next_earnings_date'].strftime('%Y-%m-%d'))
    with col3:
        st.metric("Sentiment", f"{sentiment_results['sentiment']} {sentiment_results['sentiment_emoji']}")
    
    # Results Tables
    st.subheader("Quarterly Results")
    st.dataframe(revenue_results['result_df'].style.format({
        'Close': '${:,.2f}',
        'Price_Growth': '{:.2%}',
        'Revenue_Growth': '{:.2%}'
    }))
    
    # Visualizations
    create_visualizations(revenue_results, sentiment_results)

if __name__ == "__main__":
    main()
