import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Expanded universe of stocks for more data
def get_expanded_symbols():
    """Get expanded list of liquid stocks"""
    return [
        # Mega Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK',
        # Consumer
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'UNP',
        # Energy
        'XOM', 'CVX', 'COP',
        # Communication
        'VZ', 'T', 'DIS',
        # ETFs
        'SPY', 'QQQ', 'IWM'
    ]

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch OHLCV data for a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data for {symbol}")
            return None
            
        # Clean column names
        df.columns = [col.lower() for col in df.columns]
        df['symbol'] = symbol
        
        # Add sector info (simplified)
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
        finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA']
        
        if symbol in tech_stocks:
            df['sector'] = 'Technology'
        elif symbol in finance_stocks:
            df['sector'] = 'Financial Services'
        else:
            df['sector'] = 'Other'
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_vix_data(start_date, end_date):
    """Fetch VIX data for market context"""
    try:
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(start=start_date, end=end_date)
        vix_data.columns = [col.lower() for col in vix_data.columns]
        return vix_data['close'].rename('vix')
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return None

def identify_gaps(df, gap_threshold=0.02):
    """Identify gap events in price data"""
    # Calculate previous close
    df['prev_close'] = df['close'].shift(1)
    
    # Calculate gap percentage
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Identify gaps above threshold
    df['is_gap'] = abs(df['gap_pct']) > gap_threshold
    df['gap_direction'] = np.where(df['gap_pct'] > 0, 'up', 'down')
    
    # Volume ratio (current vs 20-day average)
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    return df

def calculate_technical_indicators(df):
    """Calculate RSI, moving averages, and other indicators"""
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # Distance from moving average
    df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']
    
    # Enhanced features
    df['price_volatility_10d'] = df['close'].pct_change().rolling(10).std()
    df['price_volatility_20d'] = df['close'].pct_change().rolling(20).std()
    
    # Bollinger Bands
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['ma_20'] + (2 * bb_std)
    df['bb_lower'] = df['ma_20'] - (2 * bb_std)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df

def calculate_gap_fill_target(df, days_ahead=3):
    """Calculate if gap fills within specified days"""
    targets = []
    
    for i in range(len(df)):
        if not df.iloc[i]['is_gap']:
            targets.append(np.nan)
            continue
            
        gap_direction = df.iloc[i]['gap_direction']
        prev_close = df.iloc[i]['prev_close']
        
        # Look ahead up to 'days_ahead' days
        filled = False
        for j in range(1, min(days_ahead + 1, len(df) - i)):
            future_idx = i + j
            
            if gap_direction == 'up':
                # For up gaps, check if low reaches within 0.5% of prev_close
                if df.iloc[future_idx]['low'] <= prev_close * 1.005:
                    filled = True
                    break
            else:
                # For down gaps, check if high reaches within 0.5% of prev_close
                if df.iloc[future_idx]['high'] >= prev_close * 0.995:
                    filled = True
                    break
        
        targets.append(filled)
    
    df['gap_filled_3d'] = targets
    return df

def add_enhanced_features(df):
    """Add enhanced features for better prediction"""
    # Gap size relative to recent volatility
    df['gap_vs_volatility'] = abs(df['gap_pct']) / (df['price_volatility_10d'] + 0.001)
    
    # Volume features
    df['volume_spike'] = df['volume_ratio'] > 2.0
    df['volume_drought'] = df['volume_ratio'] < 0.5
    
    # Technical extremes
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    
    # Trend strength
    df['strong_uptrend'] = df['price_vs_ma20'] > 0.10
    df['strong_downtrend'] = df['price_vs_ma20'] < -0.10
    
    # Gap categories
    df['large_gap'] = abs(df['gap_pct']) > 0.05
    df['huge_gap'] = abs(df['gap_pct']) > 0.10
    
    return df

# Main execution
if __name__ == "__main__":
    print("ENHANCED GAP DATA COLLECTION")
    print("="*50)
    
    # Get expanded symbol list
    symbols = get_expanded_symbols()
    print(f"Collecting data for {len(symbols)} symbols...")
    
    # Extended date range for more data
    start_date = "2019-01-01"  # 6 years of data
    end_date = "2024-12-31"
    
    # Collect VIX data first
    print("Fetching VIX data...")
    vix_data = get_vix_data(start_date, end_date)
    
    # Collect stock data
    stock_data = {}
    processed_data = []
    
    for symbol in symbols:
        print(f"Fetching and processing {symbol}...")
        
        # Fetch data
        df = fetch_stock_data(symbol, start_date, end_date)
        if df is None:
            continue
        
        # Add VIX data
        if vix_data is not None:
            df = df.join(vix_data, how='left')
            df['vix'] = df['vix'].fillna(method='ffill')
        else:
            df['vix'] = 20.0  # Default VIX
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Identify gaps
        df = identify_gaps(df)
        
        # Add enhanced features
        df = add_enhanced_features(df)
        
        # Calculate gap fill targets
        df = calculate_gap_fill_target(df)
        
        # Store processed data
        processed_data.append(df)
        stock_data[symbol] = df
    
    # Combine all data
    print("Combining all data...")
    all_data = pd.concat(processed_data, ignore_index=True)
    
    # Filter for gaps only
    gap_events = all_data[all_data['is_gap'] == True].copy()
    
    print(f"\nENHANCED DATASET SUMMARY:")
    print(f"="*40)
    print(f"Total trading days: {len(all_data):,}")
    print(f"Total gap events: {len(gap_events):,}")
    print(f"Gap fill rate: {gap_events['gap_filled_3d'].mean():.2%}")
    print(f"Large gaps (>5%): {sum(gap_events['large_gap']):,}")
    print(f"Huge gaps (>10%): {sum(gap_events['huge_gap']):,}")
    
    # Breakdown by direction
    print(f"\nGap direction breakdown:")
    print(gap_events['gap_direction'].value_counts())
    
    # Breakdown by sector
    print(f"\nGaps by sector:")
    print(gap_events['sector'].value_counts())
    
    # Save enhanced dataset
    gap_events.to_csv('gap_events_enhanced.csv', index=False)
    print(f"\n✅ Enhanced gap events saved to 'gap_events_enhanced.csv'")
    print(f"Dataset expanded from 693 to {len(gap_events):,} events!")
    
    if len(gap_events) > 1500:
        print("🚀 Excellent! This dataset should give much better ML performance!")
    elif len(gap_events) > 1000:
        print("✅ Good dataset size for improved ML training!")
    else:
        print("⚠️  Dataset is larger but may need more symbols or longer timeframe")