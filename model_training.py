import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CompleteGapFillPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.label_encoders = {}
        
    def load_best_available_data(self):
        """Load the best available dataset (enhanced first, then original)"""
        datasets_to_try = [
            ('gap_events_enhanced.csv', 'ENHANCED'),
            ('gap_events.csv', 'ORIGINAL')
        ]
        
        for filename, dataset_type in datasets_to_try:
            try:
                gap_data = pd.read_csv(filename, index_col=0)
                
                # Try to convert index to datetime
                try:
                    gap_data.index = pd.to_datetime(gap_data.index)
                except:
                    gap_data.reset_index(drop=True, inplace=True)
                
                print(f"✅ Loaded {len(gap_data)} gap events from {dataset_type} dataset ({filename})")
                return gap_data, dataset_type
                
            except FileNotFoundError:
                print(f"⚠️ {filename} not found, trying next option...")
                continue
        
        print("❌ No gap events data found!")
        print("Please run data collection first:")
        print("  python simple_enhanced_data_collection.py")
        return None, None
        
    def prepare_enhanced_features(self, df):
        """Prepare comprehensive feature set"""
        features_df = df.copy()
        
        # Remove rows with missing target values
        features_df = features_df.dropna(subset=['gap_filled_3d'])
        print(f"After removing missing targets: {len(features_df)} rows")
        
        # Fill missing values intelligently
        print("Filling missing values...")
        
        # VIX handling
        if features_df['vix'].isna().all():
            print("Warning: All VIX values are null. Using default VIX = 20")
            features_df['vix'] = 20.0
        else:
            features_df['vix'] = features_df['vix'].fillna(features_df['vix'].median())
        
        # Other missing values
        features_df['rsi'] = features_df['rsi'].fillna(50)  # Neutral RSI
        features_df['volume_ratio'] = features_df['volume_ratio'].fillna(1.0)  # Normal volume
        features_df['price_vs_ma20'] = features_df['price_vs_ma20'].fillna(0.0)  # At MA
        
        # Core gap features
        features_df['gap_size_abs'] = abs(features_df['gap_pct'])
        features_df['gap_size_squared'] = features_df['gap_pct'] ** 2
        features_df['gap_size_log'] = np.log(features_df['gap_size_abs'] + 0.001)
        
        # Enhanced volatility features (if available)
        if 'price_volatility_10d' in features_df.columns:
            features_df['gap_vs_volatility'] = features_df['gap_size_abs'] / (features_df['price_volatility_10d'] + 0.001)
        else:
            # Create basic volatility estimate
            features_df['gap_vs_volatility'] = features_df['gap_size_abs'] / 0.02  # Default volatility
        
        # Volume features
        features_df['volume_ratio_log'] = np.log(features_df['volume_ratio'] + 0.1)
        features_df['volume_spike'] = features_df['volume_ratio'] > 2.0
        features_df['volume_drought'] = features_df['volume_ratio'] < 0.5
        
        # Technical indicator features
        features_df['rsi_oversold'] = features_df['rsi'] < 30
        features_df['rsi_overbought'] = features_df['rsi'] > 70
        features_df['rsi_extreme'] = (features_df['rsi'] < 25) | (features_df['rsi'] > 75)
        
        # Trend features
        features_df['strong_uptrend'] = features_df['price_vs_ma20'] > 0.10
        features_df['strong_downtrend'] = features_df['price_vs_ma20'] < -0.10
        
        # Market context features
        features_df['market_stress'] = features_df['vix'] > 30
        features_df['market_euphoria'] = features_df['vix'] < 12
        features_df['high_vix'] = features_df['vix'] > 25
        
        # Gap type features
        features_df['large_gap'] = features_df['gap_size_abs'] > 0.05
        features_df['huge_gap'] = features_df['gap_size_abs'] > 0.10
        features_df['large_up_gap'] = (features_df['gap_direction'] == 'up') & (features_df['gap_size_abs'] > 0.05)
        features_df['large_down_gap'] = (features_df['gap_direction'] == 'down') & (features_df['gap_size_abs'] > 0.05)
        
        # Time features
        features_df['day_of_week'] = 2  # Default Wednesday
        features_df['monday_gap'] = False
        features_df['friday_gap'] = False
        
        # Sector features (if available)
        if 'sector' in features_df.columns:
            features_df['tech_stock'] = features_df['sector'].isin(['Technology', 'tech'])
            features_df['finance_stock'] = features_df['sector'].isin(['Financial Services', 'Financial', 'finance'])
        else:
            features_df['tech_stock'] = False
            features_df['finance_stock'] = False
        
        # Feature interactions
        features_df['gap_size_x_volume'] = features_df['gap_size_abs'] * features_df['volume_ratio']
        features_df['rsi_x_trend'] = features_df['rsi'] * features_df['price_vs_ma20']
        features_df['vix_x_gap'] = features_df['vix'] * features_df['gap_size_abs']
        
        # Create enhanced categorical features
        features_df = self.create_categorical_features(features_df)
        
        # Define comprehensive feature set
        self.feature_cols = [
            # Core gap features
            'gap_pct', 'gap_size_abs', 'gap_size_squared', 'gap_size_log',
            'gap_vs_volatility',
            
            # Volume features
            'volume_ratio', 'volume_ratio_log', 'volume_spike', 'volume_drought',
            
            # Technical indicators
            'rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_extreme',
            'price_vs_ma20', 'strong_uptrend', 'strong_downtrend',
            
            # Market context
            'vix', 'market_stress', 'market_euphoria', 'high_vix',
            
            # Gap type features
            'large_gap', 'huge_gap', 'large_up_gap', 'large_down_gap',
            
            # Time features
            'day_of_week', 'monday_gap', 'friday_gap',
            
            # Stock type features
            'tech_stock', 'finance_stock',
            
            # Interactions
            'gap_size_x_volume', 'rsi_x_trend', 'vix_x_gap',
            
            # Categorical features
            'gap_size_category', 'vix_level', 'rsi_category', 
            'volume_category', 'trend_category', 'gap_direction'
        ]
        
        # Encode categorical variables
        categorical_cols = ['gap_size_category', 'vix_level', 'rsi_category', 
                           'volume_category', 'trend_category', 'gap_direction']
        
        for col in categorical_cols:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Remove features that don't exist and handle missing values
        existing_features = [col for col in self.feature_cols if col in features_df.columns]
        self.feature_cols = existing_features
        
        print(f"Using {len(self.feature_cols)} features: {self.feature_cols[:10]}...")
        
        # Final cleanup
        features_df = features_df.dropna(subset=self.feature_cols)
        print(f"Final dataset size: {len(features_df)} rows")
        
        return features_df
    
    def create_categorical_features(self, df):
        """Create categorical features with proper binning"""
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
    
    def train_complete_model(self, df):
        """Train the complete model with enhanced features"""
        print("Preparing enhanced features...")
        prepared_df = self.prepare_enhanced_features(df)
        
        if len(prepared_df) == 0:
            print("❌ ERROR: No data left after feature preparation!")
            return None
        
        # Prepare X and y
        X = prepared_df[self.feature_cols]
        y = prepared_df['gap_filled_3d'].astype(int)
        
        print(f"Training on {len(X)} gap events with {len(self.feature_cols)} features")
        print(f"Gap fill rate: {y.mean():.2%}")
        
        # Enhanced train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train enhanced LightGBM model
        self.model = lgb.LGBMClassifier(
            objective='binary',
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        print("Training enhanced model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Comprehensive evaluation
        return self.evaluate_complete_model(X_train, X_test, y_train, y_test)
    
    def evaluate_complete_model(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Accuracy scores
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        # AUC scores
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"\n🎯 COMPLETE MODEL PERFORMANCE:")
        print(f"Training Accuracy: {train_acc:.3f} | Training AUC: {train_auc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f} | Test AUC: {test_auc:.3f}")
        print(f"Baseline (majority class): {y_test.mean():.3f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, test_pred))
        
        # Confidence analysis at multiple thresholds
        thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        print(f"\n🔥 CONFIDENCE ANALYSIS:")
        
        best_threshold = 0.7
        best_accuracy = 0
        
        for threshold in thresholds:
            high_conf_mask = (test_proba > threshold) | (test_proba < (1 - threshold))
            if high_conf_mask.sum() > 0:
                high_conf_acc = (y_test[high_conf_mask] == test_pred[high_conf_mask]).mean()
                print(f"Confidence > {threshold:.0%}: {high_conf_mask.sum():3d} samples, {high_conf_acc:.1%} accuracy")
                
                # Track best performing threshold
                if high_conf_acc > best_accuracy and high_conf_mask.sum() >= 20:
                    best_accuracy = high_conf_acc
                    best_threshold = threshold
        
        print(f"\n⭐ BEST THRESHOLD: {best_threshold:.0%} with {best_accuracy:.1%} accuracy")
        
        # Feature importance
        self.show_feature_importance()
        
        return {
            'X_test': X_test,
            'y_test': y_test, 
            'test_pred': test_pred,
            'test_proba': test_proba,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy
        }
    
    def show_feature_importance(self):
        """Show top feature importance"""
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
        for _, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.1f}")
        
        return feature_importance_df
    
    def save_complete_model(self, results, dataset_type):
        """Save the complete model with metadata"""
        print(f"\n💾 SAVING COMPLETE MODEL...")
        
        model_data = {
            'model': self.model,
            'predictor': self,
            'feature_cols': self.feature_cols,
            'label_encoders': self.label_encoders,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': dataset_type,
            'training_samples': len(results['X_test']) * 5,  # Approximate total samples
            'test_accuracy': results['test_accuracy'],
            'test_auc': results['test_auc'],
            'best_threshold': results['best_threshold'],
            'best_accuracy': results['best_accuracy'],
            'feature_count': len(self.feature_cols)
        }
        
        # Save with multiple names for compatibility
        filenames = ['gap_fill_model.pkl', 'improved_gap_model.pkl', 'complete_gap_model.pkl']
        
        for filename in filenames:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"✅ Model saved as:")
        for filename in filenames:
            print(f"  - {filename}")
        
        # Save model summary
        summary = f"""
GAP-FILL MODEL TRAINING SUMMARY
===============================
Date: {model_data['training_date']}
Dataset: {dataset_type}
Training Samples: {model_data['training_samples']}
Features: {model_data['feature_count']}
Test Accuracy: {results['test_accuracy']:.1%}
Test AUC: {results['test_auc']:.3f}
Best Threshold: {results['best_threshold']:.0%}
Best Accuracy: {results['best_accuracy']:.1%}

Model Status: READY FOR TRADING
Next Step: python updated_daily_gap_scanner.py
"""
        
        with open('model_training_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"📄 Training summary saved to 'model_training_summary.txt'")
        
        return model_data

def run_complete_training():
    """Run the complete training pipeline"""
    print("="*70)
    print("COMPLETE GAP-FILL MODEL TRAINING SYSTEM")
    print("="*70)
    
    # Initialize predictor
    predictor = CompleteGapFillPredictor()
    
    # Load best available data
    gap_data, dataset_type = predictor.load_best_available_data()
    
    if gap_data is None:
        return None
    
    # Train complete model
    results = predictor.train_complete_model(gap_data)
    
    if results is None:
        print("❌ Model training failed!")
        return None
    
    # Save model
    model_data = predictor.save_complete_model(results, dataset_type)
    
    # Success summary
    print(f"\n" + "="*70)
    print("✅ COMPLETE MODEL TRAINING SUCCESSFUL!")
    print("="*70)
    print(f"📊 Dataset: {dataset_type} ({len(gap_data)} events)")
    print(f"🎯 Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"🔥 Best Confidence Accuracy: {results['best_accuracy']:.1%}")
    print(f"⚡ Features: {len(predictor.feature_cols)}")
    print(f"\n🚀 READY FOR LIVE TRADING!")
    print(f"Run: python updated_daily_gap_scanner.py")
    
    return model_data

if __name__ == "__main__":
    # Run complete training
    model_data = run_complete_training()
    
    if model_data:
        print(f"\n🎖️ MODEL TRAINING COMPLETE - READY TO TRADE! 🎖️")
    else:
        print(f"\n❌ MODEL TRAINING FAILED - CHECK DATA AND TRY AGAIN")