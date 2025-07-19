import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    def __init__(self, data_path='data/weekly_merged_data.csv', models_path='time_series_models.pkl'):
        """Initialize the time series predictor with data"""
        self.data_path = data_path
        self.models_path = models_path
        self.df = None
        self.models = {}
        self.feature_columns = ['CPIAUCSL', 'Brent_Oil', 'USD_Index', 'SP500', 'Gold_Price']
        self.load_data()
        self.load_or_fit_models()
        
    def load_data(self):
        """Load and prepare the data for time series analysis"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df = self.df.dropna()
            
            # Convert index to datetime if it's not already
            if 'index' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['index'])
                self.df = self.df.sort_values('Date')
            else:
                self.df['Date'] = pd.date_range(start='2010-01-01', periods=len(self.df), freq='W')
            
            self.df.set_index('Date', inplace=True)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def load_or_fit_models(self):
        """Load existing models or fit new ones if they don't exist"""
        if os.path.exists(self.models_path):
            try:
                print(f"Loading existing models from {self.models_path}...")
                with open(self.models_path, 'rb') as f:
                    self.models = pickle.load(f)
                print(f"✅ Loaded {len(self.models)} models successfully")
                return
            except Exception as e:
                print(f"⚠️ Error loading models: {e}. Will fit new models.")
        
        # Fit new models if loading failed or file doesn't exist
        print("Fitting new ARIMA models...")
        self.fit_arima_models()
        self.save_models()
    
    def check_stationarity(self, series, feature_name):
        """Check if a time series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        is_stationary = result[1] <= 0.05
        print(f"{feature_name}: {'Stationary' if is_stationary else 'Non-stationary'} (p-value: {result[1]:.4f})")
        return is_stationary
    
    def find_best_arima_order(self, series, feature_name, max_p=3, max_d=2, max_q=3):
        """Find the best ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Test different ARIMA orders
        for p in range(0, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"{feature_name} - Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit_arima_models(self):
        """Fit ARIMA models for all features"""
        print("FITTING ARIMA MODELS")
        
        for feature in self.feature_columns:
            if feature in self.df.columns:
                series = self.df[feature]
                
                # Check stationarity
                is_stationary = self.check_stationarity(series, feature)
                
                # Find best ARIMA order
                best_order = self.find_best_arima_order(series, feature)
                
                # Fit the model
                try:
                    model = ARIMA(series, order=best_order)
                    fitted_model = model.fit()
                    self.models[feature] = fitted_model
                    print(f"{feature} model fitted successfully")
                except Exception as e:
                    print(f"Error fitting {feature} model: {e}")
                    # Fallback to simple ARIMA(1,1,1)
                    try:
                        model = ARIMA(series, order=(1, 1, 1))
                        fitted_model = model.fit()
                        self.models[feature] = fitted_model
                        print(f"{feature} model fitted with fallback ARIMA(1,1,1)")
                    except Exception as e2:
                        print(f"Failed to fit {feature} model even with fallback: {e2}")
        
        print(f"\nModels fitted: {list(self.models.keys())}")
    
    def save_models(self):
        """Save the fitted models to disk"""
        try:
            with open(self.models_path, 'wb') as f:
                pickle.dump(self.models, f)
            print(f"✅ Models saved to {self.models_path}")
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def force_refit_models(self):
        """Force refitting of all models (useful when data is updated)"""
        print("🔄 Force refitting all ARIMA models...")
        self.models = {}
        self.fit_arima_models()
        self.save_models()
    
    def predict_features(self, months_ahead):
        """Predict all features for the specified number of months ahead"""
        if not self.models:
            print("No models fitted. Please run fit_arima_models() first.")
            return None
        
        # Convert months to weeks (approximate)
        weeks_ahead = int(months_ahead * 4.33)  # Average weeks per month
        
        predictions = {}
        prediction_dates = []
        
        # Generate future dates
        last_date = self.df.index[-1]
        for i in range(1, weeks_ahead + 1):
            prediction_dates.append(last_date + pd.Timedelta(weeks=i))
        
        print(f"PREDICTING FEATURES FOR {months_ahead} MONTHS AHEAD")
        
        for feature in self.feature_columns:
            if feature in self.models:
                try:
                    # Make prediction
                    forecast = self.models[feature].forecast(steps=weeks_ahead)
                    predictions[feature] = forecast.values
                    print(f"{feature}: Predicted {len(forecast)} values")
                except Exception as e:
                    print(f"Error predicting {feature}: {e}")
                    # Use last known value as fallback
                    last_value = self.df[feature].iloc[-1]
                    predictions[feature] = [last_value] * weeks_ahead
                    print(f"{feature}: Using last known value as fallback")
        
        # Create prediction dataframe
        pred_df = pd.DataFrame(predictions, index=prediction_dates)
        
        return pred_df
    
    def create_enhanced_features(self, historical_data, future_predictions):
        """Create enhanced features for gold price prediction using historical and predicted data"""
        enhanced_features = []
        
        for i, (date, row) in enumerate(future_predictions.iterrows()):
            features = {}
            
            # Base features from predictions
            features.update({
                'CPIAUCSL': row['CPIAUCSL'],
                'Brent_Oil': row['Brent_Oil'],
                'USD_Index': row['USD_Index'],
                'SP500': row['SP500']
            })
            
            # Lag features (use historical data for first few predictions)
            if i == 0:
                # First prediction - use last historical values
                features.update({
                    'CPIAUCSL_lag1': historical_data['CPIAUCSL'].iloc[-1],
                    'Brent_Oil_lag1': historical_data['Brent_Oil'].iloc[-1],
                    'USD_Index_lag1': historical_data['USD_Index'].iloc[-1],
                    'SP500_lag1': historical_data['SP500'].iloc[-1],
                    'CPIAUCSL_lag2': historical_data['CPIAUCSL'].iloc[-2],
                    'Brent_Oil_lag2': historical_data['Brent_Oil'].iloc[-2],
                    'USD_Index_lag2': historical_data['USD_Index'].iloc[-2],
                    'SP500_lag2': historical_data['SP500'].iloc[-2]
                })
            elif i == 1:
                # Second prediction - use first prediction as lag1
                features.update({
                    'CPIAUCSL_lag1': future_predictions.iloc[0]['CPIAUCSL'],
                    'Brent_Oil_lag1': future_predictions.iloc[0]['Brent_Oil'],
                    'USD_Index_lag1': future_predictions.iloc[0]['USD_Index'],
                    'SP500_lag1': future_predictions.iloc[0]['SP500'],
                    'CPIAUCSL_lag2': historical_data['CPIAUCSL'].iloc[-1],
                    'Brent_Oil_lag2': historical_data['Brent_Oil'].iloc[-1],
                    'USD_Index_lag2': historical_data['USD_Index'].iloc[-1],
                    'SP500_lag2': historical_data['SP500'].iloc[-1]
                })
            else:
                # Use previous predictions as lags
                features.update({
                    'CPIAUCSL_lag1': future_predictions.iloc[i-1]['CPIAUCSL'],
                    'Brent_Oil_lag1': future_predictions.iloc[i-1]['Brent_Oil'],
                    'USD_Index_lag1': future_predictions.iloc[i-1]['USD_Index'],
                    'SP500_lag1': future_predictions.iloc[i-1]['SP500'],
                    'CPIAUCSL_lag2': future_predictions.iloc[i-2]['CPIAUCSL'],
                    'Brent_Oil_lag2': future_predictions.iloc[i-2]['Brent_Oil'],
                    'USD_Index_lag2': future_predictions.iloc[i-2]['USD_Index'],
                    'SP500_lag2': future_predictions.iloc[i-2]['SP500']
                })
            
            # Moving averages (use historical data for initial calculations)
            if i < 4:
                # Use historical data for MA4
                features.update({
                    'CPIAUCSL_ma4': historical_data['CPIAUCSL'].tail(4).mean(),
                    'Brent_Oil_ma4': historical_data['Brent_Oil'].tail(4).mean(),
                    'USD_Index_ma4': historical_data['USD_Index'].tail(4).mean(),
                    'SP500_ma4': historical_data['SP500'].tail(4).mean()
                })
            else:
                # Use recent predictions for MA4
                recent_cpi = list(future_predictions.iloc[i-4:i]['CPIAUCSL'])
                recent_oil = list(future_predictions.iloc[i-4:i]['Brent_Oil'])
                recent_usd = list(future_predictions.iloc[i-4:i]['USD_Index'])
                recent_sp = list(future_predictions.iloc[i-4:i]['SP500'])
                
                features.update({
                    'CPIAUCSL_ma4': np.mean(recent_cpi),
                    'Brent_Oil_ma4': np.mean(recent_oil),
                    'USD_Index_ma4': np.mean(recent_usd),
                    'SP500_ma4': np.mean(recent_sp)
                })
            
            if i < 12:
                # Use historical data for MA12
                features.update({
                    'CPIAUCSL_ma12': historical_data['CPIAUCSL'].tail(12).mean(),
                    'Brent_Oil_ma12': historical_data['Brent_Oil'].tail(12).mean(),
                    'USD_Index_ma12': historical_data['USD_Index'].tail(12).mean(),
                    'SP500_ma12': historical_data['SP500'].tail(12).mean()
                })
            else:
                # Use recent predictions for MA12
                recent_cpi_12 = list(future_predictions.iloc[i-12:i]['CPIAUCSL'])
                recent_oil_12 = list(future_predictions.iloc[i-12:i]['Brent_Oil'])
                recent_usd_12 = list(future_predictions.iloc[i-12:i]['USD_Index'])
                recent_sp_12 = list(future_predictions.iloc[i-12:i]['SP500'])
                
                features.update({
                    'CPIAUCSL_ma12': np.mean(recent_cpi_12),
                    'Brent_Oil_ma12': np.mean(recent_oil_12),
                    'USD_Index_ma12': np.mean(recent_usd_12),
                    'SP500_ma12': np.mean(recent_sp_12)
                })
            
            # Gold momentum and volatility (use historical data)
            if 'Gold_Price' in historical_data.columns:
                gold_prices = historical_data['Gold_Price']
                features.update({
                    'Gold_momentum_ma4': gold_prices.pct_change().tail(4).mean(),
                    'Gold_volatility': gold_prices.tail(12).std()
                })
            else:
                features.update({
                    'Gold_momentum_ma4': 0.0,
                    'Gold_volatility': 100.0
                })
            
            # Interaction features
            features.update({
                'Oil_USD_interaction': row['Brent_Oil'] * row['USD_Index'],
                'CPI_SP500_interaction': row['CPIAUCSL'] * row['SP500']
            })
            
            enhanced_features.append(features)
        
        return pd.DataFrame(enhanced_features, index=future_predictions.index)
    
    def predict_gold_price(self, enhanced_features, model, scaler, feature_names):
        """Predict gold prices using the enhanced features and trained model"""
        try:
            # Ensure correct column order
            feature_df = enhanced_features[feature_names]
            
            # Scale features
            features_scaled = scaler.transform(feature_df)
            
            # Make predictions
            predictions = model.predict(features_scaled)
            
            return predictions
        except Exception as e:
            print(f"Error predicting gold price: {e}")
            return None
    
    def plot_predictions(self, historical_data, feature_predictions, gold_predictions, months_ahead):
        """Plot the predictions for all features"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Time Series Predictions - {months_ahead} Months Ahead', fontsize=16)
        
        features = ['CPIAUCSL', 'Brent_Oil', 'USD_Index', 'SP500', 'Gold_Price']
        
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            
            # Plot historical data
            axes[row, col].plot(historical_data.index, historical_data[feature], 
                              label='Historical', color='blue', linewidth=2)
            
            # Plot predictions
            if feature in feature_predictions.columns:
                axes[row, col].plot(feature_predictions.index, feature_predictions[feature], 
                                  label='Predicted', color='red', linewidth=2, linestyle='--')
            
            # Plot gold price predictions separately
            if feature == 'Gold_Price' and gold_predictions is not None:
                axes[row, col].plot(feature_predictions.index, gold_predictions, 
                                  label='Gold Price Predicted', color='gold', linewidth=2, linestyle='-')
            
            axes[row, col].set_title(f'{feature} Prediction')
            axes[row, col].set_xlabel('Date')
            axes[row, col].set_ylabel(feature)
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove the last subplot if we have 5 features
        if len(features) == 5:
            axes[1, 2].remove()
        
        plt.tight_layout()
        return fig
    
    def run_full_prediction(self, months_ahead, model, scaler, feature_names):
        """Run the complete prediction pipeline"""
        print(f"\n" + "="*60)
        print(f"RUNNING FULL PREDICTION PIPELINE - {months_ahead} MONTHS")
        print("="*60)
        
        # Step 1: Check if models are loaded, if not, they should be loaded in __init__
        if not self.models:
            print("⚠️ No models available. Fitting new models...")
            self.fit_arima_models()
            self.save_models()
        
        # Step 2: Predict all features
        feature_predictions = self.predict_features(months_ahead)
        
        if feature_predictions is None:
            print("❌ Failed to predict features")
            return None
        
        # Step 3: Create enhanced features
        enhanced_features = self.create_enhanced_features(self.df, feature_predictions)
        
        # Step 4: Predict gold prices
        gold_predictions = self.predict_gold_price(enhanced_features, model, scaler, feature_names)
        
        if gold_predictions is None:
            print("❌ Failed to predict gold prices")
            return None
        
        # Step 5: Create results dataframe
        results = feature_predictions.copy()
        results['Gold_Price_Predicted'] = gold_predictions
        
        print(f"✅ Successfully predicted {len(results)} weeks ({months_ahead} months)")
        print(f"📅 Prediction period: {results.index[0].strftime('%Y-%m-%d')} to {results.index[-1].strftime('%Y-%m-%d')}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = TimeSeriesPredictor()
    
    # Load the trained model
    try:
        with open('ridge_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        print("✅ Ridge model loaded successfully")
        
        # Run prediction for 6 months
        print("\n🔮 Running prediction for 6 months...")
        results = predictor.run_full_prediction(6, model, scaler, feature_names)
        
        if results is not None:
            print(f"\n📊 Prediction completed!")
            print(f"📅 Predicted {len(results)} weeks")
            print(f"💰 Gold price range: ${results['Gold_Price_Predicted'].min():.2f} - ${results['Gold_Price_Predicted'].max():.2f}")
            print(f"📈 Final predicted gold price: ${results['Gold_Price_Predicted'].iloc[-1]:.2f}")
            
            # Save sample results
            results.to_csv('sample_time_series_predictions.csv')
            print("💾 Sample predictions saved to 'sample_time_series_predictions.csv'")
        
    except FileNotFoundError:
        print("❌ Ridge model file not found. Please run predict_model.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Time series predictor setup completed!")
