# 💰 Gold Price Prediction Model

A machine learning application that predicts gold prices using economic indicators (CPI, Brent Oil, USD Index, S&P 500) with Ridge Regression and time series analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gold-prediction-ml.git
   cd gold-prediction-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   ```bash
   python weekly_data_merge.py
   ```

4. **Train the model**
   ```bash
   python predict_model.py
   ```

5. **Initialize time series models**
   ```bash
   python initialize_time_series.py
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Features

### 🤖 Machine Learning Model
- **Ridge Regression** with feature engineering
- **Time Series Analysis** using ARIMA models
- **Feature Engineering**: Lag features, moving averages, momentum, volatility
- **Model Performance**: 77.05% R² accuracy with 5.57% average error

### 📈 Prediction Capabilities
- **Single Prediction**: Real-time gold price prediction
- **Time Series Prediction**: Up to 12 months ahead forecasting
- **Batch Prediction**: Process multiple data points at once
- **Interactive Visualizations**: Plotly charts and correlation analysis

### 🎯 Economic Indicators Used
- **CPI (Consumer Price Index)**: Inflation indicator
- **Brent Oil Price**: Energy market indicator
- **USD Index**: Currency strength indicator
- **S&P 500**: Stock market indicator

## 🏗️ Project Structure

```
gold-prediction-ml/
├── data/                          # Data files
│   ├── CPIAUCSL_daily.csv        # CPI data
│   ├── DCOILBRENTEU.csv          # Brent oil data
│   ├── DTWEXBGS.csv              # USD index data
│   ├── SP500.csv                 # S&P 500 data
│   ├── Gold_price_averages_in_a range_of_currencies_since_1978.xlsx
│   └── weekly_merged_data.csv    # Merged weekly data
├── streamlit_app.py              # Main Streamlit application
├── predict_model.py              # Model training script
├── time_series.py                # Time series prediction module
├── weekly_data_merge.py          # Data preprocessing script
├── initialize_time_series.py     # Time series model initialization
├── test_time_series.py           # Testing script
├── ridge_model.pkl               # Trained Ridge model
├── time_series_models.pkl        # Trained ARIMA models
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Usage

### Web Application
The main interface is a Streamlit web app with four tabs:

1. **Single Prediction**: Enter current economic indicators for instant gold price prediction
2. **Time Series Prediction**: Forecast gold prices up to 12 months ahead
3. **Batch Prediction**: Upload CSV files for multiple predictions
4. **Data Visualizations**: Explore relationships between indicators and gold prices

### Command Line Usage

**Train the model:**
```bash
python predict_model.py
```

**Test time series functionality:**
```bash
python test_time_series.py
```

**Initialize time series models:**
```bash
python initialize_time_series.py
```

## 📊 Model Performance

- **R² Score**: 77.05%
- **Mean Absolute Percentage Error**: 5.57%
- **Root Mean Square Error**: ~$85
- **Cross-validation**: 5-fold time series split

## 🎯 Key Features

### Feature Engineering
- **Lag Features**: Previous week's values (lag1, lag2)
- **Moving Averages**: 4-week and 12-week averages
- **Momentum Indicators**: Price change rates
- **Volatility Measures**: Rolling standard deviation
- **Interaction Features**: Cross-indicator relationships

### Time Series Analysis
- **ARIMA Models**: For each economic indicator
- **Stationarity Testing**: Augmented Dickey-Fuller test
- **Auto-parameter Selection**: AIC-based model selection
- **Multi-step Forecasting**: Up to 12 months ahead

## 📈 Data Sources

- **CPI Data**: Federal Reserve Economic Data (FRED)
- **Brent Oil**: Energy Information Administration (EIA)
- **USD Index**: Federal Reserve Economic Data (FRED)
- **S&P 500**: Yahoo Finance
- **Gold Prices**: Historical averages from multiple sources

## 🛠️ Technical Stack

- **Python**: 3.8+
- **Machine Learning**: scikit-learn, statsmodels
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Web Framework**: Streamlit
- **Time Series**: ARIMA models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. The predictions are based on historical data and machine learning models, which do not guarantee future performance. This should not be considered as financial advice. Always consult with financial professionals before making investment decisions.

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/gold-prediction-ml/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🎉 Acknowledgments

- Federal Reserve Economic Data (FRED) for economic indicators
- Energy Information Administration (EIA) for oil price data
- Yahoo Finance for market data
- The open-source community for the amazing libraries used in this project

---

**Made with ❤️ for the financial data science community** 