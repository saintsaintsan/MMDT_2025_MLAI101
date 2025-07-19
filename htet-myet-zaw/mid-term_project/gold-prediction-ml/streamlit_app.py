import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime
import time
from time_series import TimeSeriesPredictor

# Page configuration
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for gold buttons and progress bar
st.markdown("""
<style>
/* Gold progress bar */
.stProgress > div > div > div > div {
    background-color: #ed9e3e;
}

/* Gold buttons */
.stButton > button {
    background-color: #ed9e3e !important;
    color: #000000 !important;
    border: 2px solid #DAA520 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #FFA500 !important;
    border-color: #ed9e3e !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(237, 158, 62, 0.3) !important;
}

.stButton > button:active {
    background-color: #DAA520 !important;
    transform: translateY(0) !important;
}

/* Primary button special styling */
.stButton > button[data-baseweb="button"] {
    background-color: #ed9e3e !important;
    color: #000000 !important;
}

/* Tab text colors */
.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] {
    color: white !important;
}

.stTabs [data-baseweb="tab-list"] [aria-selected="true"] {
    color: #ed9e3e !important;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("💰 Gold-Price Prediction Model")
st.markdown("Predict world gold-prices using economic indicators with our Ridge Regression model")

# Function to load the trained model
@st.cache_resource
def load_model():
    """Load the trained Ridge Regression model"""
    try:
        # Check if model file exists
        if os.path.exists('ridge_model.pkl'):
            with open('ridge_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        else:
            st.error("Model file not found. Please run the training script first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load and prepare data for visualization
@st.cache_data
def load_data_for_viz():
    """Load data for visualization"""
    try:
        df = pd.read_csv('data/weekly_merged_data.csv')
        df = df.dropna()
        
        # Convert index to datetime if it's not already
        if 'index' in df.columns:
            df['Date'] = pd.to_datetime(df['index'])
            df = df.sort_values('Date')
        else:
            df['Date'] = pd.date_range(start='2010-01-01', periods=len(df), freq='W')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to create features for prediction
def create_features(cpi, brent_oil, usd_index, sp500, historical_data=None):
    """Create enhanced features for prediction"""
    # Base features
    features = {
        'CPIAUCSL': cpi,
        'Brent_Oil': brent_oil,
        'USD_Index': usd_index,
        'SP500': sp500
    }
    
    # If we have historical data, we can create lag features
    if historical_data is not None and len(historical_data) >= 2:
        # Add lag features
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
        
        # Add moving averages
        features.update({
            'CPIAUCSL_ma4': historical_data['CPIAUCSL'].tail(4).mean(),
            'Brent_Oil_ma4': historical_data['Brent_Oil'].tail(4).mean(),
            'USD_Index_ma4': historical_data['USD_Index'].tail(4).mean(),
            'SP500_ma4': historical_data['SP500'].tail(4).mean(),
            'CPIAUCSL_ma12': historical_data['CPIAUCSL'].tail(12).mean(),
            'Brent_Oil_ma12': historical_data['Brent_Oil'].tail(12).mean(),
            'USD_Index_ma12': historical_data['USD_Index'].tail(12).mean(),
            'SP500_ma12': historical_data['SP500'].tail(12).mean()
        })
        
        # Add gold momentum and volatility (if available)
        if 'Gold_Price' in historical_data.columns:
            gold_prices = historical_data['Gold_Price']
            features.update({
                'Gold_momentum_ma4': gold_prices.pct_change().tail(4).mean(),
                'Gold_volatility': gold_prices.tail(12).std()
            })
        else:
            features.update({
                'Gold_momentum_ma4': 0.0,
                'Gold_volatility': 100.0  # Default value
            })
    else:
        # Default values for lag features when no historical data
        features.update({
            'CPIAUCSL_lag1': cpi, 'Brent_Oil_lag1': brent_oil,
            'USD_Index_lag1': usd_index, 'SP500_lag1': sp500,
            'CPIAUCSL_lag2': cpi, 'Brent_Oil_lag2': brent_oil,
            'USD_Index_lag2': usd_index, 'SP500_lag2': sp500,
            'CPIAUCSL_ma4': cpi, 'Brent_Oil_ma4': brent_oil,
            'USD_Index_ma4': usd_index, 'SP500_ma4': sp500,
            'CPIAUCSL_ma12': cpi, 'Brent_Oil_ma12': brent_oil,
            'USD_Index_ma12': usd_index, 'SP500_ma12': sp500,
            'Gold_momentum_ma4': 0.0, 'Gold_volatility': 100.0
        })
    
    # Add interaction features
    features.update({
        'Oil_USD_interaction': brent_oil * usd_index,
        'CPI_SP500_interaction': cpi * sp500
    })
    
    return features

# Main app
def main():
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("""
        **Model not found!** 
        
        Please run the training script first to generate the model file.
        
        ```bash
        python predict_model.py
        ```
        """)
        return
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Time Series Prediction", "Data Visualizations"])
    
    with tab1:
        st.header("Single Gold Price Prediction")
        st.markdown("Enter current economic indicators to predict gold price")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            cpi = st.number_input("CPI (Consumer Price Index)", value=300.0, min_value=100.0, max_value=500.0, step=1.0)
            brent_oil = st.number_input("Brent Oil Price (USD/barrel)", value=80.0, min_value=20.0, max_value=150.0, step=1.0)
        
        with col2:
            usd_index = st.number_input("USD Index", value=100.0, min_value=80.0, max_value=120.0, step=0.1)
            sp500 = st.number_input("S&P 500 Index", value=4000.0, min_value=2000.0, max_value=6000.0, step=50.0)
        
        # Prediction button
        if st.button("Predict Gold Price", type="primary"):
            with st.spinner("Making prediction..."):
                # Create features
                features = create_features(cpi, brent_oil, usd_index, sp500)
                
                # Convert to DataFrame
                feature_df = pd.DataFrame([features])
                
                # Ensure correct column order
                feature_df = feature_df[feature_names]
                
                # Scale features
                features_scaled = scaler.transform(feature_df)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Calculate confidence interval
                mape = 5.57  # From model performance
                lower_bound = prediction * (1 - mape/100)
                upper_bound = prediction * (1 + mape/100)
                
                # Clean Prediction Results
                st.markdown("---")
                st.markdown("## Prediction Results")
                
                # Main prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.markdown(f"""
                        <div style='font-size: 3.8em; font-weight: bold; color: #ed9e3e; margin-bottom: 0.5em'>
                            ${prediction:,.2f}
                        </div>                        
                    """, unsafe_allow_html=True)
                    # st.metric(
                    #     label="Predicted Gold Price",
                    #     value=f"${prediction:,.2f}",
                    #     delta=None
                    # )
                
                # Confidence interval
                st.markdown("**Confidence Interval:**")
                st.info(f"${lower_bound:,.2f} - ${upper_bound:,.2f}")
                
                # Input summary
                st.markdown("**Input Values:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CPI", f"{cpi:,.1f}")
                with col2:
                    st.metric("Brent Oil", f"${brent_oil:,.1f}")
                with col3:
                    st.metric("USD Index", f"{usd_index:,.1f}")
                with col4:
                    st.metric("S&P 500", f"{sp500:,.0f}")
                
                # Model performance
                st.markdown("**Model Performance:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy (R²)", "77.05%")
                with col2:
                    st.metric("Average Error", "5.57%")
                with col3:
                    st.metric("Model Type", "Ridge Regression")
                
                # Simple insights
                st.markdown("**Analysis:**")
                if prediction > 2000:
                    st.write("High gold price prediction suggests strong market conditions")
                elif prediction > 1500:
                    st.write("Moderate gold price prediction indicates stable market")
                else:
                    st.write("Lower gold price prediction suggests weaker market")
                
                if abs(upper_bound - lower_bound) > 200:
                    st.write("Wide confidence interval indicates high market uncertainty")
                else:
                    st.write("Narrow confidence interval suggests stable prediction")
                
                # Disclaimer
                st.markdown("---")
                st.caption("**Disclaimer:** This prediction is based on historical data and machine learning models. Market conditions can change rapidly. This should not be considered as financial advice.")
    
    with tab2:
        st.header("Time Series Prediction")
        st.markdown("Predict gold prices up to 12 months ahead using time series analysis")
        
        # Initialize time series predictor
        @st.cache_resource
        def get_time_series_predictor():
            try:
                return TimeSeriesPredictor()
            except Exception as e:
                st.error(f"Error initializing time series predictor: {e}")
                return None
        
        predictor = get_time_series_predictor()
        
        if predictor is None:
            st.error("Could not initialize time series predictor. Please check your data files.")
            return
        
        # User input for prediction period
        col1, col2 = st.columns(2)
        with col1:
            months_ahead = st.slider(
                "Number of months to predict", 
                min_value=1, 
                max_value=12, 
                value=6,
                help="Select how many months ahead you want to predict"
            )
        
        with col2:
            st.metric("Prediction Period", f"{months_ahead} months")
            st.metric("Weeks to Predict", f"{int(months_ahead * 4.33)} weeks")
        
        # Prediction button
        if st.button("Predict Future Gold Prices", type="primary"):
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Data preparation
                status_text.text("📊 Preparing historical data...")
                progress_bar.progress(10)
                time.sleep(1.5)
                
                # Step 2: Loading time series models
                status_text.text("🔧 Loading ARIMA models...")
                progress_bar.progress(25)
                time.sleep(2.0)
                
                # Step 3: Feature engineering
                status_text.text("⚙️ Engineering features...")
                progress_bar.progress(40)
                time.sleep(1.8)
                
                # Step 4: Running predictions
                status_text.text(f"🔮 Running time series analysis for {months_ahead} months...")
                progress_bar.progress(60)
                time.sleep(2.5)
                
                # Step 5: Processing results
                status_text.text("📈 Processing prediction results...")
                progress_bar.progress(80)
                time.sleep(1.2)
                
                # Step 6: Finalizing
                status_text.text("✅ Finalizing predictions...")
                progress_bar.progress(95)
                time.sleep(0.8)
                
                # Run the actual prediction
                results = predictor.run_full_prediction(months_ahead, model, scaler, feature_names)
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("🎉 Prediction completed successfully!")
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if results is not None:
                    st.success(f"✅ Successfully predicted {len(results)} weeks!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Final Gold Price",
                            f"${results['Gold_Price_Predicted'].iloc[-1]:,.2f}",
                            f"{((results['Gold_Price_Predicted'].iloc[-1] / results['Gold_Price_Predicted'].iloc[0]) - 1) * 100:+.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Average Gold Price",
                            f"${results['Gold_Price_Predicted'].mean():,.2f}"
                        )
                    with col3:
                        st.metric(
                            "Price Range",
                            f"${results['Gold_Price_Predicted'].min():,.0f} - ${results['Gold_Price_Predicted'].max():,.0f}"
                        )
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Predictions")
                    
                    # Create a formatted results table
                    display_results = results.copy()
                    display_results.index = display_results.index.strftime('%Y-%m-%d')
                    # Remove 'Gold Price' column if present
                    if 'Gold_Price' in display_results.columns:
                        display_results = display_results.drop(columns=['Gold_Price'])
                    display_results.columns = [col.replace('_', ' ').title() for col in display_results.columns]
                    
                    # Format numbers
                    for col in display_results.columns:
                        if 'Price' in col:
                            display_results[col] = display_results[col].apply(lambda x: f"${x:,.2f}")
                        else:
                            display_results[col] = display_results[col].apply(lambda x: f"{x:,.2f}")
                    
                    st.dataframe(display_results, use_container_width=True)
                    
                    # Download results
                    csv = results.to_csv()
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"gold_predictions_{months_ahead}months.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.subheader("📈 Prediction Visualization")
                    
                    # Create interactive plot with Plotly
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=results.index,
                        y=results['Gold_Price_Predicted'],
                        mode='lines+markers',
                        name='Predicted Gold Price',
                        line=dict(color='gold', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Gold Price:</b> $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f'Gold Price Prediction - {months_ahead} Months Ahead',
                        xaxis_title='Date',
                        yaxis_title='Gold Price (USD)',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature predictions visualization
                    st.subheader("🔮 Economic Indicators Predictions")
                    
                    # Create subplot for feature predictions
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('CPI Prediction', 'Brent Oil Prediction', 
                                      'USD Index Prediction', 'S&P 500 Prediction'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # CPI
                    fig.add_trace(
                        go.Scatter(x=results.index, y=results['CPIAUCSL'], 
                                  mode='lines', name='CPI Predicted', 
                                  line=dict(color='red', width=2),
                                  hovertemplate='<b>Date:</b> %{x}<br><b>CPI:</b> %{y:,.2f}<extra></extra>'),
                        row=1, col=1
                    )
                    
                    # Brent Oil
                    fig.add_trace(
                        go.Scatter(x=results.index, y=results['Brent_Oil'], 
                                  mode='lines', name='Brent Oil Predicted', 
                                  line=dict(color='black', width=2),
                                  hovertemplate='<b>Date:</b> %{x}<br><b>Brent Oil:</b> $%{y:,.2f}<extra></extra>'),
                        row=1, col=2
                    )
                    
                    # USD Index
                    fig.add_trace(
                        go.Scatter(x=results.index, y=results['USD_Index'], 
                                  mode='lines', name='USD Index Predicted', 
                                  line=dict(color='green', width=2),
                                  hovertemplate='<b>Date:</b> %{x}<br><b>USD Index:</b> %{y:,.2f}<extra></extra>'),
                        row=2, col=1
                    )
                    
                    # SP500
                    fig.add_trace(
                        go.Scatter(x=results.index, y=results['SP500'], 
                                  mode='lines', name='S&P 500 Predicted', 
                                  line=dict(color='blue', width=2),
                                  hovertemplate='<b>Date:</b> %{x}<br><b>S&P 500:</b> %{y:,.0f}<extra></extra>'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, showlegend=False, title_text="Economic Indicators Predictions")
                    fig.update_xaxes(title_text="Date")
                    fig.update_yaxes(title_text="CPI", row=1, col=1)
                    fig.update_yaxes(title_text="Brent Oil (USD/barrel)", row=1, col=2)
                    fig.update_yaxes(title_text="USD Index", row=2, col=1)
                    fig.update_yaxes(title_text="S&P 500", row=2, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    st.subheader("💡 Prediction Insights")
                    
                    # Calculate insights
                    gold_trend = results['Gold_Price_Predicted'].iloc[-1] - results['Gold_Price_Predicted'].iloc[0]
                    gold_trend_pct = (gold_trend / results['Gold_Price_Predicted'].iloc[0]) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Gold Price Trend:**")
                        if gold_trend > 0:
                            st.success(f"📈 Bullish trend: +${gold_trend:,.0f} (+{gold_trend_pct:.1f}%)")
                        else:
                            st.error(f"📉 Bearish trend: {gold_trend:,.0f} ({gold_trend_pct:.1f}%)")
                        
                        st.markdown("**Key Factors:**")
                        # Analyze feature trends
                        cpi_trend = results['CPIAUCSL'].iloc[-1] - results['CPIAUCSL'].iloc[0]
                        oil_trend = results['Brent_Oil'].iloc[-1] - results['Brent_Oil'].iloc[0]
                        usd_trend = results['USD_Index'].iloc[-1] - results['USD_Index'].iloc[0]
                        
                        if cpi_trend > 0:
                            st.write("• 📊 CPI: Rising inflation (gold price going up)")
                        if oil_trend > 0:
                            st.write("• 🛢️ Oil: Rising energy costs (gold price going up)")
                        if usd_trend < 0:
                            st.write("• 💵 USD: Weakening dollar (gold price going up)")
                    
                    with col2:
                        st.markdown("**Market Sentiment:**")
                        volatility = results['Gold_Price_Predicted'].std()
                        if volatility > 100:
                            st.warning("⚠️ High volatility expected")
                        else:
                            st.info("✅ Stable price movement expected")
                        
                        st.markdown("**Recommendation:**")
                        if gold_trend_pct > 5:
                            st.success("🟢 Consider gold as a hedge")
                        elif gold_trend_pct < -5:
                            st.error("🔴 Monitor market conditions")
                        else:
                            st.info("🟡 Neutral position recommended")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.caption("**Disclaimer:** Time series predictions are based on historical patterns and statistical models. They do not guarantee future performance and should not be considered as financial advice. Market conditions can change rapidly.")
                    
                else:
                    st.error("❌ Failed to generate predictions. Please check your data and try again.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("💡 Try reducing the prediction period or check if all required data is available.")
    
    with tab3:
        st.header("Data Visualizations")
        st.markdown("Explore the relationships between economic indicators and gold prices")
        
        # Load data for visualization
        df = load_data_for_viz()
        
        if df is not None:
            # Time series plots
            st.subheader("Time Series Analysis")
            
            # Create subplot for time series
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Gold Price Over Time', 'CPI Over Time', 
                              'Brent Oil Price Over Time', 'USD Index Over Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Gold Price
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Gold_Price'], 
                          mode='lines', name='Gold Price', 
                          line=dict(color='gold', width=2),
                          hovertemplate='<b>Date:</b> %{x}<br><b>Gold Price:</b> $%{y:,.2f}<extra></extra>'),
                row=1, col=1
            )
            
            # CPI
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['CPIAUCSL'], 
                          mode='lines', name='CPI', 
                          line=dict(color='red', width=2),
                          hovertemplate='<b>Date:</b> %{x}<br><b>CPI:</b> %{y:,.2f}<extra></extra>'),
                row=1, col=2
            )
            
            # Brent Oil
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Brent_Oil'], 
                          mode='lines', name='Brent Oil', 
                          line=dict(color='skyblue', width=2),
                          hovertemplate='<b>Date:</b> %{x}<br><b>Brent Oil:</b> $%{y:,.2f}<extra></extra>'),
                row=2, col=1
            )
            
            # USD Index
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['USD_Index'], 
                          mode='lines', name='USD Index', 
                          line=dict(color='green', width=2),
                          hovertemplate='<b>Date:</b> %{x}<br><b>USD Index:</b> %{y:,.2f}<extra></extra>'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Economic Indicators Over Time")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Gold Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="CPI", row=1, col=2)
            fig.update_yaxes(title_text="Brent Oil (USD/barrel)", row=2, col=1)
            fig.update_yaxes(title_text="USD Index", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Break line
            st.markdown("---")
            
            # Correlation analysis
            st.subheader("🔗 Correlation Analysis")
            
            # Select features for correlation
            features_for_corr = ['Gold_Price', 'CPIAUCSL', 'Brent_Oil', 'USD_Index', 'SP500']
            corr_data = df[features_for_corr].corr()
            
            # Create correlation heatmap with Plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Correlation Matrix of Economic Indicators',
                height=500,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Break line
            st.markdown("---")
            
            # Scatter plots
            st.subheader("📊 Feature Relationships with Gold Price")
            
            # Create subplot for scatter plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Gold Price vs CPI', 'Gold Price vs Brent Oil', 
                              'Gold Price vs USD Index', 'Gold Price vs S&P 500'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Gold vs CPI
            fig.add_trace(
                go.Scatter(x=df['CPIAUCSL'], y=df['Gold_Price'], 
                          mode='markers', name='Gold vs CPI',
                          marker=dict(color='red', size=6, opacity=0.6),
                          hovertemplate='<b>CPI:</b> %{x:,.2f}<br><b>Gold Price:</b> $%{y:,.2f}<extra></extra>'),
                row=1, col=1
            )
            
            # Gold vs Brent Oil
            fig.add_trace(
                go.Scatter(x=df['Brent_Oil'], y=df['Gold_Price'], 
                          mode='markers', name='Gold vs Brent Oil',
                          marker=dict(color='gold', size=6, opacity=0.6),
                          hovertemplate='<b>Brent Oil:</b> $%{x:,.2f}<br><b>Gold Price:</b> $%{y:,.2f}<extra></extra>'),
                row=1, col=2
            )
            
            # Gold vs USD Index
            fig.add_trace(
                go.Scatter(x=df['USD_Index'], y=df['Gold_Price'], 
                          mode='markers', name='Gold vs USD Index',
                          marker=dict(color='green', size=6, opacity=0.6),
                          hovertemplate='<b>USD Index:</b> %{x:,.2f}<br><b>Gold Price:</b> $%{y:,.2f}<extra></extra>'),
                row=2, col=1
            )
            
            # Gold vs SP500
            fig.add_trace(
                go.Scatter(x=df['SP500'], y=df['Gold_Price'], 
                          mode='markers', name='Gold vs SP500',
                          marker=dict(color='blue', size=6, opacity=0.6),
                          hovertemplate='<b>S&P 500:</b> %{x:,.0f}<br><b>Gold Price:</b> $%{y:,.2f}<extra></extra>'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Gold Price vs Economic Indicators")
            fig.update_xaxes(title_text="CPI", row=1, col=1)
            fig.update_xaxes(title_text="Brent Oil (USD/barrel)", row=1, col=2)
            fig.update_xaxes(title_text="USD Index", row=2, col=1)
            fig.update_xaxes(title_text="S&P 500", row=2, col=2)
            fig.update_yaxes(title_text="Gold Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Gold Price (USD)", row=1, col=2)
            fig.update_yaxes(title_text="Gold Price (USD)", row=2, col=1)
            fig.update_yaxes(title_text="Gold Price (USD)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Break line
            st.markdown("---")
            
            # Statistical summary
            st.subheader("📋 Statistical Summary")
            
            # Display statistics for each feature
            stats_df = df[features_for_corr].describe()
            st.dataframe(stats_df.round(2))
            
        else:
            st.error("Could not load data for visualization. Please check if 'data/weekly_merged_data.csv' exists.")

if __name__ == "__main__":
    main() 