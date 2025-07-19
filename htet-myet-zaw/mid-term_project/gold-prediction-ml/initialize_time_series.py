#!/usr/bin/env python3
"""
Initialize and save time series models for faster loading
"""

import pandas as pd
import numpy as np
from time_series import TimeSeriesPredictor
import pickle

def initialize_time_series_models():
    """Initialize and save time series models"""
    print("🚀 Initializing Time Series Models")
    print("="*50)
    
    try:
        # Initialize predictor (this will automatically fit and save models)
        print("1. Initializing TimeSeriesPredictor...")
        predictor = TimeSeriesPredictor()
        
        print("2. Checking if models were saved...")
        if hasattr(predictor, 'models') and predictor.models:
            print(f"✅ {len(predictor.models)} models are ready!")
            print(f"📁 Models saved to: {predictor.models_path}")
            
            # List the models
            print("\n📋 Available models:")
            for feature in predictor.models.keys():
                print(f"  • {feature}")
            
            # Test a quick prediction
            print("\n3. Testing model functionality...")
            try:
                # Load the ridge model for testing
                with open('ridge_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                model = model_data['model']
                scaler = model_data['scaler']
                feature_names = model_data['feature_names']
                
                # Test prediction for 1 month
                results = predictor.run_full_prediction(1, model, scaler, feature_names)
                
                if results is not None:
                    print("✅ Test prediction successful!")
                    print(f"📊 Predicted {len(results)} weeks")
                    print(f"💰 Sample gold price: ${results['Gold_Price_Predicted'].iloc[0]:.2f}")
                else:
                    print("❌ Test prediction failed")
                    
            except FileNotFoundError:
                print("⚠️ Ridge model not found. Time series models are ready, but you need to run predict_model.py first for full functionality.")
            except Exception as e:
                print(f"⚠️ Test prediction error: {e}")
            
            print("\n🎉 Time series models are ready for use!")
            print("💡 You can now run the Streamlit app with fast model loading.")
            
        else:
            print("❌ Models were not properly initialized")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

if __name__ == "__main__":
    success = initialize_time_series_models()
    if success:
        print("\n✅ Initialization completed successfully!")
    else:
        print("\n💥 Initialization failed. Please check the error messages above.") 