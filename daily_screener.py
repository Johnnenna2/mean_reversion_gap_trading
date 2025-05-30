import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from datetime import datetime, timedelta, time
import pytz
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import argparse
import os
import requests
import json
warnings.filterwarnings('ignore')

class CompleteGapFillPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.label_encoders = {}

class DiscordNotificationManager:
    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.bot_name = "Gap Trading Bot"
        
    def send_recommendations(self, recommendations_df, gaps_df=None):
        """Send trading recommendations to Discord"""
        
        if recommendations_df is None or recommendations_df.empty:
            self.send_no_opportunities_found(gaps_df)
            return
        
        # Create rich embed message
        embed = self.create_recommendations_embed(recommendations_df)
        
        # Send to Discord
        self.send_webhook(embed=embed)
        
        # If many recommendations, send summary stats as separate message
        if len(recommendations_df) > 3:
            self.send_summary_stats(recommendations_df)
    
    def create_recommendations_embed(self, df):
        """Create a rich Discord embed for recommendations"""
        
        # Sort by confidence
        df_sorted = df.sort_values('confidence', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
        
        # Main embed
        embed = {
            "title": "🎯 Gap Trading Opportunities Found!",
            "description": f"**{len(df)} trading opportunities** detected by the ML model",
            "color": 0x00ff00,  # Green color
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Gap Trading Bot • ML-Powered Analysis",
                "icon_url": "https://cdn.discordapp.com/embed/avatars/0.png"
            },
            "fields": []
        }
        
        # Add individual recommendations as fields
        for i, (_, rec) in enumerate(df_sorted.iterrows()):
            confidence_pct = float(rec['confidence'].rstrip('%'))
            
            # Emoji and color based on confidence
            if confidence_pct >= 90:
                emoji = "🔥"
                confidence_color = "**"
            elif confidence_pct >= 85:
                emoji = "⚡"
                confidence_color = "**"
            elif confidence_pct >= 80:
                emoji = "💡"
                confidence_color = ""
            else:
                emoji = "📊"
                confidence_color = ""
            
            # Action emoji
            action_emoji = "📉" if "PUTS" in rec['action'] else "📈"
            
            field_name = f"{emoji} {rec['symbol']} ({rec['sector']}) {action_emoji}"
            
            field_value = f"""
            **{rec['action']}** • {confidence_color}{rec['confidence']}{confidence_color} confidence
            💰 Price: {rec['current_price']} • 📊 Gap: {rec['gap_pct']}
            📈 Volume: {rec['volume_ratio']} • 🎯 RSI: {rec['rsi']}
            📏 **{rec['position_size']}**
            """
            
            embed["fields"].append({
                "name": field_name,
                "value": field_value.strip(),
                "inline": False
            })
            
            # Limit to 5 recommendations per embed (Discord limit is 25 fields)
            if i >= 4:
                break
        
        return embed
    
    def send_summary_stats(self, df):
        """Send summary statistics as a separate message"""
        
        high_conf = sum(df['confidence'].str.rstrip('%').astype(float) >= 85)
        ultra_high_conf = sum(df['confidence'].str.rstrip('%').astype(float) >= 90)
        
        # Sector breakdown
        sector_counts = df['sector'].value_counts()
        sector_text = "\n".join([f"• {sector}: {count}" for sector, count in sector_counts.items()])
        
        # Action breakdown
        puts_count = sum(df['action'].str.contains('PUTS'))
        calls_count = sum(df['action'].str.contains('CALLS'))
        
        embed = {
            "title": "📊 Analysis Summary",
            "color": 0x0099ff,  # Blue color
            "fields": [
                {
                    "name": "🎯 Confidence Breakdown",
                    "value": f"🔥 Ultra-high (≥90%): {ultra_high_conf}\n⚡ High (≥85%): {high_conf}\n📊 Total opportunities: {len(df)}",
                    "inline": True
                },
                {
                    "name": "📈 Strategy Breakdown", 
                    "value": f"📉 Put plays: {puts_count}\n📈 Call plays: {calls_count}",
                    "inline": True
                },
                {
                    "name": "🏢 Sectors",
                    "value": sector_text,
                    "inline": False
                }
            ]
        }
        
        self.send_webhook(embed=embed)
    
    def send_no_opportunities_found(self, gaps_df=None):
        """Send message when no trading opportunities are found"""
        
        # Check if any gaps were found at all
        gaps_found = len(gaps_df) if gaps_df is not None and not gaps_df.empty else 0
        
        if gaps_found > 0:
            description = f"Found {gaps_found} gaps, but none met our confidence threshold for trading"
            color = 0xffaa00  # Orange
            emoji = "⚠️"
        else:
            description = "No significant gaps (>1.8%) detected in today's market"
            color = 0x808080  # Gray
            emoji = "📭"
        
        embed = {
            "title": f"{emoji} No Trading Opportunities Today",
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Gap Trading Bot • Next scan in a few hours"
            }
        }
        
        if gaps_found > 0 and gaps_df is not None:
            # Show the gaps that were found but didn't make the cut
            gap_list = []
            for _, gap in gaps_df.iterrows():
                gap_list.append(f"• {gap['symbol']}: {gap['gap_pct']:+.1%}")
            
            embed["fields"] = [{
                "name": "🔍 Gaps Detected (Below Threshold)",
                "value": "\n".join(gap_list[:10]),  # Limit to 10
                "inline": False
            }]
        
        self.send_webhook(embed=embed)
    
    def send_error(self, error_message):
        """Send error notification to Discord"""
        embed = {
            "title": "🚨 Gap Scanner Error",
            "description": f"```\n{error_message}\n```",
            "color": 0xff0000,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Gap Trading Bot • Error Alert"
            }
        }
        
        self.send_webhook(embed=embed)
    
    def send_scan_start(self):
        """Send notification when scan starts"""
        embed = {
            "title": "🤖 Gap Scanner Starting",
            "description": "Scanning market for gap trading opportunities...",
            "color": 0x00ffff,  # Cyan
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_webhook(embed=embed)
    
    def send_webhook(self, content=None, embed=None):
        """Send message to Discord webhook"""
        
        if not self.webhook_url:
            print("⚠️ Discord webhook URL not configured")
            return False
        
        payload = {
            "username": self.bot_name,
        }
        
        if content:
            payload["content"] = content
        
        if embed:
            payload["embeds"] = [embed]
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 204:
                print("✅ Discord notification sent successfully")
                return True
            else:
                print(f"❌ Discord webhook failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Discord webhook error: {str(e)}")
            return False

class PerfectDailyGapScanner:
    def __init__(self, model_file=None):
        self.model = None
        self.predictor = None
        self.model_data = None
        self.universe = self.get_trained_universe()
        self.est = pytz.timezone('US/Eastern')
        
        if model_file:
            self.load_model(model_file)
    
    def get_trained_universe(self):
        """Get universe from actual training data to ensure compatibility"""
        try:
            gap_data = pd.read_csv('gap_events_enhanced.csv')
            trained_tickers = sorted(gap_data['symbol'].unique())
            print(f"Using {len(trained_tickers)} tickers from training data")
            return trained_tickers
        except FileNotFoundError:
            print("Enhanced data not found, using fallback universe")
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK',
                'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE',
                'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE'
            ]
    
    def is_optimal_scan_time(self):
        """Check if current time is optimal for gap scanning"""
        now_est = datetime.now(self.est)
        current_time = now_est.time()
        
        # Pre-market: 4:00 AM - 9:30 AM (OPTIMAL)
        premarket_start = time(4, 0)
        premarket_end = time(9, 30)
        
        # After hours: 4:00 PM - 8:00 PM (GOOD)
        afterhours_start = time(16, 0)
        afterhours_end = time(20, 0)
        
        # Market hours: 9:30 AM - 4:00 PM (ACCEPTABLE for automated runs)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if premarket_start <= current_time <= premarket_end:
            return "OPTIMAL", "Pre-market - best time for gap scanning"
        elif afterhours_start <= current_time <= afterhours_end:
            return "GOOD", "After hours - good for gap scanning"
        elif market_open <= current_time <= market_close:
            return "ACCEPTABLE", "Market hours - acceptable for automated scanning"
        else:
            return "ACCEPTABLE", "Overnight - acceptable for gap scanning"
    
    def validate_scan_timing(self, automated=False):
        """Validate timing and warn user if problematic"""
        quality, message = self.is_optimal_scan_time()
        
        print(f"🕐 SCAN TIMING: {quality}")
        print(f"   {message}")
        
        # In automated mode, don't prompt user - just proceed
        if automated:
            return True
        
        if quality == "PROBLEMATIC":
            print("⚠️  WARNING: Scanning during market hours!")
            print("   - Data may be incomplete or delayed")
            print("   - Gap calculations may be inaccurate")
            print("   - Recommended: Run pre-market (4-9:30 AM) or after hours (4-8 PM)")
            print()
            
            response = input("Continue with potentially unreliable data? (y/n): ").lower()
            if response != 'y':
                print("Scanner stopped. Please run during optimal hours.")
                return False
        elif quality in ["OPTIMAL", "GOOD", "ACCEPTABLE"]:
            if not automated:
                print("✅ Good timing for gap scanning")
        
        print()
        return True
    
    def get_sector_classification(self, symbol):
        """Classify symbols by sector"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 
                          'QCOM', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'NOW', 'TEAM', 'SNOW', 'PLTR'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'BRK-B', 'AFRM', 'SOFI'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'GILD', 'BIIB', 'MRNA'],
            'Consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'TGT', 'COST', 'SBUX'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY'],
            'Transportation': ['UBER', 'LYFT', 'ABNB', 'AAL', 'DAL', 'UAL'],
            'EV/Clean': ['TSLA', 'F', 'GM', 'RIVN', 'NIO', 'ENPH', 'PLUG'],
            'ETF': ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP']
        }
        
        for sector, symbols in sectors.items():
            if symbol in symbols:
                return sector
        return 'Other'
    
    def validate_data_quality(self, df, symbol):
        """Comprehensive data quality validation"""
        
        # Check 1: Sufficient data points
        if len(df) < 2:
            return False, 'Insufficient data points'
        
        # Check 2: Recent data (within last 5 trading days)
        latest_date = df.index[-1]
        days_old = (datetime.now() - latest_date).days
        if days_old > 5:
            return False, f'Data too old ({days_old} days)'
        
        # Check 3: No zero/negative prices
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        price_checks = [
            latest_row['open'], latest_row['high'], 
            latest_row['low'], latest_row['close'],
            prev_row['close']
        ]
        
        if any(price <= 0 for price in price_checks):
            return False, 'Zero or negative prices detected'
        
        # Check 4: Logical price relationships
        if not (latest_row['low'] <= latest_row['open'] <= latest_row['high'] and
                latest_row['low'] <= latest_row['close'] <= latest_row['high']):
            return False, 'Illogical OHLC relationships'
        
        # Check 5: Reasonable volume
        if latest_row['volume'] <= 0:
            return False, 'Zero or negative volume'
        
        # Check 6: No extreme price movements (>50% in one day - likely data error)
        daily_change = abs((latest_row['close'] - prev_row['close']) / prev_row['close'])
        if daily_change > 0.5:
            return False, f'Extreme price movement ({daily_change:.1%})'
        
        return True, 'All checks passed'
    
    def fetch_current_data(self, symbol, days_back=60):
        """Fetch recent data with robust validation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"❌ {symbol}: No data returned")
                return None
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            df = df.dropna()
            
            # Remove timezone information to avoid mixing tz-aware and tz-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Validate data quality
            is_valid, reason = self.validate_data_quality(df, symbol)
            if not is_valid:
                print(f"❌ {symbol}: Data validation failed - {reason}")
                return None
            
            # Add metadata
            df['symbol'] = symbol
            df['sector'] = self.get_sector_classification(symbol)
            
            return df
            
        except Exception as e:
            print(f"❌ {symbol}: Error fetching data - {e}")
            return None
    
    def calculate_gap_with_debug(self, data, symbol, show_debug=True):
        """Calculate gap with detailed debugging output"""
        if len(data) < 2:
            return None
            
        latest_day = data.iloc[-1]
        prev_day = data.iloc[-2]
        
        # Calculate gap
        gap_pct = (latest_day['open'] - prev_day['close']) / prev_day['close']
        
        # Debug output for significant gaps
        if abs(gap_pct) > 0.015 and show_debug:  # 1.5% threshold for debug output
            print(f"\n🔍 GAP DEBUG - {symbol}:")
            print(f"   Previous day ({prev_day.name.date()}): Close = ${prev_day['close']:.2f}")
            print(f"   Current day ({latest_day.name.date()}): Open = ${latest_day['open']:.2f}")
            print(f"   Gap: ${latest_day['open'] - prev_day['close']:+.2f} ({gap_pct:+.1%})")
            
            # Validation warning for extreme gaps
            if abs(gap_pct) > 0.2:  # 20% gap is suspicious
                print(f"   ⚠️  EXTREME GAP WARNING - Verify this data manually!")
        
        return gap_pct
    
    def scan_for_gaps(self, gap_threshold=0.02, min_volume=50000, automated=False):
        """Scan for gaps with comprehensive validation and debugging"""
        
        # Validate scan timing
        if not self.validate_scan_timing(automated=automated):
            return pd.DataFrame()
        
        print(f"📊 SCANNING {len(self.universe)} SYMBOLS FOR GAPS:")
        print(f"   Gap threshold: {gap_threshold:.1%}")
        print(f"   Min volume: {min_volume:,}")
        print("=" * 80)
        
        gaps_found = []
        vix_data = self.get_vix_data()
        
        processed = 0
        errors = 0
        data_issues = []
        
        for symbol in self.universe:
            print(f"📈 {symbol}...", end=' ')
            
            try:
                data = self.fetch_current_data(symbol)
                if data is None:
                    errors += 1
                    data_issues.append(f"{symbol}: Data fetch failed")
                    print("❌")
                    continue
                
                # Add VIX data
                if vix_data is not None:
                    data = data.join(vix_data, how='left')
                    data['vix'] = data['vix'].fillna(method='ffill')
                else:
                    data['vix'] = 20.0
                
                # Calculate technical indicators
                data = self.calculate_indicators(data)
                
                # Calculate gap with debugging
                gap_pct = self.calculate_gap_with_debug(data, symbol, show_debug=not automated)
                if gap_pct is None:
                    errors += 1
                    print("❌")
                    continue
                
                latest_day = data.iloc[-1]
                prev_day = data.iloc[-2]
                
                # Filter criteria
                if (abs(gap_pct) > gap_threshold and 
                    latest_day['volume'] > min_volume):
                    
                    gap_info = {
                        'symbol': symbol,
                        'sector': latest_day['sector'],
                        'date': latest_day.name,
                        'gap_pct': gap_pct,
                        'gap_direction': 'up' if gap_pct > 0 else 'down',
                        'gap_size_abs': abs(gap_pct),
                        'current_price': latest_day['close'],
                        'volume': latest_day['volume'],
                        'volume_ratio': latest_day.get('volume_ratio', 1.0),
                        'rsi': latest_day.get('rsi', 50.0),
                        'price_vs_ma20': latest_day.get('price_vs_ma20', 0.0),
                        'vix': latest_day.get('vix', 20.0),
                        'prev_close': prev_day['close'],
                        'open_price': latest_day['open'],
                        'high': latest_day['high'],
                        'low': latest_day['low']
                    }
                    
                    # Enhanced gap features
                    volatility = data['price_volatility_10d'].iloc[-1] if 'price_volatility_10d' in data.columns else 0.02
                    gap_info['gap_vs_volatility'] = abs(gap_pct) / (volatility + 0.001)
                    gap_info['large_gap'] = abs(gap_pct) > 0.05
                    gap_info['huge_gap'] = abs(gap_pct) > 0.10
                    
                    gaps_found.append(gap_info)
                    print(f"✅ GAP: {gap_pct:+.1%}")
                else:
                    if not automated:
                        print(f"ℹ️  ({gap_pct:+.1%})")
                    else:
                        print("ℹ️")
                
                processed += 1
                
            except Exception as e:
                errors += 1
                data_issues.append(f"{symbol}: {str(e)}")
                print("❌")
                continue
        
        # Results summary
        gaps_df = pd.DataFrame(gaps_found)
        
        print("\n" + "="*80)
        print("📊 SCAN RESULTS:")
        print("="*80)
        print(f"✅ Symbols processed: {processed}")
        print(f"❌ Errors encountered: {errors}")
        print(f"🎯 Gaps found: {len(gaps_df)}")
        
        if data_issues and not automated:
            print(f"\n⚠️  DATA ISSUES ({len(data_issues)}):")
            for issue in data_issues[:5]:  # Show first 5 issues
                print(f"   - {issue}")
            if len(data_issues) > 5:
                print(f"   ... and {len(data_issues) - 5} more")
        
        if not gaps_df.empty:
            print(f"\n📋 GAPS BY SECTOR:")
            sector_counts = gaps_df['sector'].value_counts()
            for sector, count in sector_counts.items():
                print(f"   {sector}: {count}")
        
        return gaps_df
    
    def get_vix_data(self, days_back=30):
        """Fetch recent VIX data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=start_date, end=end_date)
            vix_data.columns = [col.lower() for col in vix_data.columns]
            
            # Remove timezone information to match stock data
            if hasattr(vix_data.index, 'tz') and vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)
            
            return vix_data['close'].rename('vix')
        except Exception as e:
            print(f"⚠️  VIX data unavailable: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators matching training data"""
        # Previous close
        df['prev_close'] = df['close'].shift(1)
        
        # Volume ratio (need at least 20 days)
        if len(df) >= 20:
            df['volume_ma20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20']
        else:
            df['volume_ratio'] = 1.0  # Default
        
        # RSI (need at least 14 days)
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = 50.0  # Default neutral RSI
        
        # Moving averages (need at least 20 days)
        if len(df) >= 20:
            df['ma_20'] = df['close'].rolling(20).mean()
            df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']
        else:
            df['price_vs_ma20'] = 0.0  # Default
        
        # Enhanced volatility features (need at least 10 days)
        if len(df) >= 10:
            df['price_volatility_10d'] = df['close'].pct_change().rolling(10).std()
        else:
            df['price_volatility_10d'] = 0.02  # Default 2% volatility
        
        return df
    
    def prepare_prediction_features(self, gaps_df):
        """Prepare features exactly matching the training data"""
        if gaps_df.empty:
            return gaps_df
        
        prediction_data = gaps_df.copy()
        
        # Fill missing values
        prediction_data['vix'] = prediction_data['vix'].fillna(20.0)
        prediction_data['rsi'] = prediction_data['rsi'].fillna(50.0)
        prediction_data['volume_ratio'] = prediction_data['volume_ratio'].fillna(1.0)
        prediction_data['price_vs_ma20'] = prediction_data['price_vs_ma20'].fillna(0.0)
        
        # Core gap features
        prediction_data['gap_size_abs'] = abs(prediction_data['gap_pct'])
        prediction_data['gap_size_squared'] = prediction_data['gap_pct'] ** 2
        prediction_data['gap_size_log'] = np.log(prediction_data['gap_size_abs'] + 0.001)
        prediction_data['gap_vs_volatility'] = prediction_data.get('gap_vs_volatility', 
                                                                  prediction_data['gap_size_abs'] / 0.02)
        
        # Volume features
        prediction_data['volume_ratio_log'] = np.log(prediction_data['volume_ratio'] + 0.1)
        prediction_data['volume_spike'] = prediction_data['volume_ratio'] > 2.0
        prediction_data['volume_drought'] = prediction_data['volume_ratio'] < 0.5
        
        # Technical indicator features
        prediction_data['rsi_oversold'] = prediction_data['rsi'] < 30
        prediction_data['rsi_overbought'] = prediction_data['rsi'] > 70
        prediction_data['rsi_extreme'] = (prediction_data['rsi'] < 25) | (prediction_data['rsi'] > 75)
        
        # Trend features
        prediction_data['strong_uptrend'] = prediction_data['price_vs_ma20'] > 0.10
        prediction_data['strong_downtrend'] = prediction_data['price_vs_ma20'] < -0.10
        
        # Market context features
        prediction_data['market_stress'] = prediction_data['vix'] > 30
        prediction_data['market_euphoria'] = prediction_data['vix'] < 12
        prediction_data['high_vix'] = prediction_data['vix'] > 25
        
        # Gap type features
        prediction_data['large_gap'] = prediction_data['gap_size_abs'] > 0.05
        prediction_data['huge_gap'] = prediction_data['gap_size_abs'] > 0.10
        prediction_data['large_up_gap'] = (prediction_data['gap_direction'] == 'up') & prediction_data['large_gap']
        prediction_data['large_down_gap'] = (prediction_data['gap_direction'] == 'down') & prediction_data['large_gap']
        
        # Time features
        prediction_data['day_of_week'] = 2  # Default Wednesday
        prediction_data['monday_gap'] = False
        prediction_data['friday_gap'] = False
        
        # Stock type features
        prediction_data['tech_stock'] = prediction_data['sector'] == 'Technology'
        prediction_data['finance_stock'] = prediction_data['sector'] == 'Financial'
        
        # Feature interactions
        prediction_data['gap_size_x_volume'] = prediction_data['gap_size_abs'] * prediction_data['volume_ratio']
        prediction_data['rsi_x_trend'] = prediction_data['rsi'] * prediction_data['price_vs_ma20']
        # Feature interactions
        prediction_data['gap_size_x_volume'] = prediction_data['gap_size_abs'] * prediction_data['volume_ratio']
        prediction_data['rsi_x_trend'] = prediction_data['rsi'] * prediction_data['price_vs_ma20']
        prediction_data['vix_x_gap'] = prediction_data['vix'] * prediction_data['gap_size_abs']
        
        # Categorical features with proper encoding
        prediction_data = self.create_categorical_features(prediction_data)
        
        return prediction_data
    
    def create_categorical_features(self, df):
        """Create categorical features matching training"""
        # Gap size categories
        df['gap_size_category'] = pd.cut(
            df['gap_size_abs'],
            bins=[0, 0.015, 0.025, 0.04, 0.07, 1.0],
            labels=['tiny', 'small', 'medium', 'large', 'huge']
        )
        
        # VIX categories
        df['vix_level'] = pd.cut(
            df['vix'],
            bins=[0, 12, 18, 25, 35, 100],
            labels=['very_low', 'low', 'medium', 'high', 'extreme']
        )
        
        # RSI categories
        df['rsi_category'] = pd.cut(
            df['rsi'],
            bins=[0, 25, 35, 65, 75, 100],
            labels=['very_oversold', 'oversold', 'neutral', 'overbought', 'very_overbought']
        )
        
        # Volume categories
        df['volume_category'] = pd.cut(
            df['volume_ratio'],
            bins=[0, 0.3, 0.7, 1.5, 3.0, 100],
            labels=['very_low', 'low', 'normal', 'high', 'extreme']
        )
        
        # Trend categories
        df['trend_category'] = pd.cut(
            df['price_vs_ma20'],
            bins=[-1, -0.10, -0.03, 0.03, 0.10, 1],
            labels=['strong_down', 'weak_down', 'sideways', 'weak_up', 'strong_up']
        )
        
        return df
    
    def predict_gap_fills(self, gaps_df):
        """Make predictions using the loaded model"""
        if self.model is None or gaps_df.empty:
            print("No model loaded or no gaps to predict!")
            return gaps_df
        
        try:
            # Prepare features
            prediction_data = self.prepare_prediction_features(gaps_df)
            
            # Get feature columns from saved model
            feature_cols = self.model_data.get('feature_cols', [])
            label_encoders = self.model_data.get('label_encoders', {})
            
            # Encode categorical variables using saved encoders
            categorical_cols = ['gap_size_category', 'vix_level', 'rsi_category', 
                               'volume_category', 'trend_category', 'gap_direction']
            
            for col in categorical_cols:
                if col in prediction_data.columns and col in label_encoders:
                    try:
                        prediction_data[col] = label_encoders[col].transform(prediction_data[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        prediction_data[col] = 0
            
            # Select only features that exist in both training and prediction data
            available_features = [col for col in feature_cols if col in prediction_data.columns]
            
            if len(available_features) == 0:
                print("No matching features found!")
                return gaps_df
            
            # Make predictions
            X = prediction_data[available_features]
            
            # Fill any remaining missing values
            X = X.fillna(0)
            
            # Get predictions
            predictions = self.model.predict_proba(X)[:, 1]
            
            # Add predictions to dataframe
            prediction_data['gap_fill_probability'] = predictions
            prediction_data['gap_fill_prediction'] = predictions > 0.5
            
            print(f"✅ Made predictions using {len(available_features)} features")
            
            return prediction_data
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return gaps_df
    
    def get_trading_recommendations(self, predicted_gaps, confidence_threshold=0.75):
        """Generate trading recommendations with CORRECTED logic"""
        if 'gap_fill_probability' not in predicted_gaps.columns:
            print("No predictions available!")
            return pd.DataFrame()
        
        # Filter for high-confidence predictions
        high_confidence = predicted_gaps[
            (predicted_gaps['gap_fill_probability'] > confidence_threshold) |
            (predicted_gaps['gap_fill_probability'] < (1 - confidence_threshold))
        ].copy()
        
        if high_confidence.empty:
            print(f"No predictions with confidence > {confidence_threshold:.0%}")
            return pd.DataFrame()
        
        recommendations = []
        
        for _, row in high_confidence.iterrows():
            prob = row['gap_fill_probability']
            confidence = max(prob, 1 - prob)
            
            # Fix gap direction - convert to string if needed
            gap_dir_raw = row['gap_direction']
            if isinstance(gap_dir_raw, (int, float)):
                gap_direction = 'up' if gap_dir_raw > 0 else 'down'
            else:
                gap_direction = str(gap_dir_raw).lower()
            
            # CORRECTED LOGIC: Determine trade direction
            if prob > confidence_threshold:
                # High probability of gap fill - trade AGAINST the gap direction
                if gap_direction == 'up':
                    action = "BUY PUTS"        # Gap up → expect price to come back down
                    strategy = "Gap Fill Play"
                else:
                    action = "BUY CALLS"       # Gap down → expect price to come back up  
                    strategy = "Gap Fill Play"
            else:
                # Low probability of gap fill - trade WITH the gap direction
                if gap_direction == 'up':
                    action = "BUY CALLS"       # Gap up → expect continued upward move
                    strategy = "Gap Continuation"
                else:
                    action = "BUY PUTS"        # Gap down → expect continued downward move
                    strategy = "Gap Continuation"
            
            # Position sizing based on confidence
            if confidence > 0.9:
                position_size = "EXTRA LARGE (3%)"
            elif confidence > 0.85:
                position_size = "LARGE (2%)"
            elif confidence > 0.8:
                position_size = "MEDIUM (1.5%)"
            else:
                position_size = "SMALL (1%)"
            
            recommendations.append({
                'symbol': row['symbol'],
                'sector': row['sector'],
                'action': action,
                'strategy': strategy,
                'gap_pct': f"{row['gap_pct']:.1%}",
                'confidence': f"{confidence:.1%}",
                'position_size': position_size,
                'current_price': f"${row['current_price']:.2f}",
                'volume': f"{row['volume']:,.0f}",
                'volume_ratio': f"{row['volume_ratio']:.1f}x",
                'rsi': f"{row['rsi']:.0f}",
                'vix': f"{row['vix']:.1f}",
                'rationale': f"{strategy} - {row['gap_pct']:.1%} gap, {confidence:.0%} confidence"
            })
        
        return pd.DataFrame(recommendations).sort_values('confidence', ascending=False)
    
    def load_model(self, filename='improved_gap_model.pkl'):
        """Load trained model with proper error handling"""
        model_files = [filename, 'gap_fill_model.pkl', 'complete_gap_model.pkl']
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    self.model_data = pickle.load(f)
                
                # Extract model components
                if isinstance(self.model_data, dict):
                    self.model = self.model_data.get('model')
                    self.predictor = self.model_data.get('predictor')
                else:
                    self.model = self.model_data
                
                print(f"✅ Model loaded from {model_file}")
                return True
                
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
        
        print("❌ No model files found!")
        return False
    
    def perfect_daily_scan_and_predict(self, automated=False):
        """Complete daily scanning and prediction workflow with robust data validation"""
        print("="*80)
        print(f"ROBUST GAP SCAN & PREDICT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*80)
        
        # 1. Scan for gaps with robust validation
        gaps = self.scan_for_gaps(gap_threshold=0.018, min_volume=30000, automated=automated)
        
        if gaps.empty:
            print("\n📭 No significant gaps found today.")
            return None, None
        
        if not automated:
            print(f"\n📋 VALIDATED GAPS FOUND:")
            print("-" * 80)
            for _, gap in gaps.iterrows():
                print(f"{gap['symbol']:5} | {gap['sector']:12} | {gap['gap_pct']:+6.1%} | "
                      f"${gap['current_price']:7.2f} | Vol: {gap['volume']/1000:6.0f}K")
        
        # 2. Make predictions
        if self.model is not None:
            predicted_gaps = self.predict_gap_fills(gaps)
            
            # 3. Generate recommendations
            recommendations = self.get_trading_recommendations(predicted_gaps, confidence_threshold=0.75)
            
            if not recommendations.empty:
                if not automated:
                    print(f"\n🎯 TRADING RECOMMENDATIONS:")
                    print("="*80)
                    
                    for _, rec in recommendations.iterrows():
                        print(f"\n{rec['symbol']} ({rec['sector']}) - {rec['action']}")
                        print(f"  Gap: {rec['gap_pct']} | Confidence: {rec['confidence']}")
                        print(f"  Price: {rec['current_price']} | Volume: {rec['volume']} ({rec['volume_ratio']})")
                        print(f"  RSI: {rec['rsi']} | VIX: {rec['vix']}")
                        print(f"  Position Size: {rec['position_size']}")
                        print(f"  Strategy: {rec['rationale']}")
                
                # Save recommendations
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f'robust_gap_recommendations_{timestamp}.csv'
                recommendations.to_csv(filename, index=False)
                print(f"\n💾 Recommendations saved to {filename}")
                
                # Summary
                if not automated:
                    print(f"\n📊 SUMMARY:")
                    print(f"Total recommendations: {len(recommendations)}")
                    high_conf = sum(recommendations['confidence'].str.rstrip('%').astype(float) > 85)
                    print(f"Ultra-high confidence (>85%): {high_conf}")
                    print(f"Sectors: {recommendations['sector'].nunique()}")
                    
                    # Manual verification reminder
                    print(f"\n🔍 MANUAL VERIFICATION RECOMMENDED:")
                    print("   Check these symbols on Yahoo Finance to confirm gap data:")
                    for _, rec in recommendations.iterrows():
                        print(f"   - {rec['symbol']}: finance.yahoo.com/quote/{rec['symbol']}")
                
            else:
                print("\n⚠️ No high-confidence recommendations today.")
        else:
            print("\n❌ No model loaded - showing validated gaps only.")
            recommendations = pd.DataFrame()
        
        return gaps, recommendations

def run_perfect_scanner():
    """Main function to run the perfect scanner"""
    scanner = PerfectDailyGapScanner()
    
    # Load model
    if scanner.load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Running without model - gaps only")
    
    # Run scan
    gaps, recommendations = scanner.perfect_daily_scan_and_predict()
    
    return gaps, recommendations

def run_automated_scanner():
    """Enhanced scanner for GitHub Actions automation"""
    discord = DiscordNotificationManager()
    
    try:
        # Send start notification
        discord.send_scan_start()
        
        # Initialize scanner
        scanner = PerfectDailyGapScanner()
        
        # Load model
        if not scanner.load_model():
            discord.send_error("❌ Model loading failed! Check model files.")
            return False
        
        print("✅ Model loaded successfully for automated scan")
        
        # Run the scan
        gaps, recommendations = scanner.perfect_daily_scan_and_predict(automated=True)
        
        # Send Discord notifications
        discord.send_recommendations(recommendations, gaps)
        
        # Save results with timestamp
        if recommendations is not None and not recommendations.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f'automated_gap_recommendations_{timestamp}.csv'
            recommendations.to_csv(filename, index=False)
            print(f"💾 Results saved to {filename}")
            
            # Also save a summary log
            with open(f'scan_log_{timestamp}.txt', 'w') as f:
                f.write(f"Automated Gap Scan Results\n")
                f.write(f"=========================\n")
                f.write(f"Scan Time: {datetime.now()}\n")
                f.write(f"Gaps Found: {len(gaps) if gaps is not None else 0}\n")
                f.write(f"Recommendations: {len(recommendations)}\n")
                f.write(f"High Confidence (>85%): {sum(recommendations['confidence'].str.rstrip('%').astype(float) > 85)}\n")
                f.write(f"Sectors: {recommendations['sector'].nunique()}\n")
        
        return True
        
    except Exception as e:
        error_msg = f"Scanner failed with error: {str(e)}"
        print(f"❌ {error_msg}")
        discord.send_error(error_msg)
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Gap Trading Scanner')
    parser.add_argument('--automated', action='store_true', 
                       help='Run in automated mode for GitHub Actions')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with limited universe')
    parser.add_argument('--notify-start', action='store_true',
                       help='Send start notification only (for testing)')
    
    args = parser.parse_args()
    
    if args.notify_start:
        # Test Discord notifications
        discord = DiscordNotificationManager()
        discord.send_scan_start()
        return
    
    if args.automated:
        print("🤖 Running in AUTOMATED mode for GitHub Actions")
        success = run_automated_scanner()
        sys.exit(0 if success else 1)
    
    elif args.test:
        print("🧪 Running in TEST mode")
        scanner = PerfectDailyGapScanner()
        scanner.universe = ['AAPL', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 'META', 'SPY', 'QQQ']
        scanner.load_model()
        gaps, recs = scanner.perfect_daily_scan_and_predict()
        
        # Send test notification
        discord = DiscordNotificationManager()
        discord.send_recommendations(recs, gaps)
    
    else:
        print("🚀 Running in INTERACTIVE mode")
        gaps, recs = run_perfect_scanner()

if __name__ == "__main__":
    print("🚀 ROBUST GAP SCANNER WITH DATA VALIDATION")
    print("=" * 50)
    
    main()
    
    print(f"\n" + "="*80)
    print("ROBUST SCANNER COMPLETE!")
    print("="*80)