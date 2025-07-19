#!/usr/bin/env python3
"""
Test script for time series prediction functionality
"""

import pandas as pd
import numpy as np
import pickle
from time_series import TimeSeriesPredictor

def test_time_series_prediction():
    """Test the time series prediction functionality"""
    print("🧪 Testing Time Series Prediction")
    print("="*50)
    
    try:
        # Initialize predictor
        print("1. Initializing TimeSeriesPredictor...")
        predictor = TimeSeriesPredictor()
        print("Predictor initialized successfully")
        
        # Load the trained model
        print("\n2. Loading trained model...")
        with open('ridge_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        print("Model loaded successfully")
        
        # Test prediction for 3 months
        print("\n3. Testing prediction for 3 months...")
        results = predictor.run_full_prediction(3, model, scaler, feature_names)
        
        if results is not None:
            print("Prediction successful!")
            print(f"Predicted {len(results)} weeks")
            print(f"Date range: {results.index[0]} to {results.index[-1]}")
            print(f"Gold price range: ${results['Gold_Price_Predicted'].min():.2f} - ${results['Gold_Price_Predicted'].max():.2f}")
            
            # Show sample predictions
            print("\nSample predictions:")
            print(results.head().round(2))
            
            return True
        else:
            print("Prediction failed")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_time_series_prediction()
    if success:
        print("\n🎉 All tests passed! Time series prediction is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.") 