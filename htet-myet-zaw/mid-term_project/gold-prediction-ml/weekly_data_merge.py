import pandas as pd

# Load CPI (monthly, to be forward-filled weekly)
cpi = pd.read_csv('data/CPIAUCSL_daily.csv')
cpi['date'] = pd.to_datetime(cpi['date'])
cpi = cpi.set_index('date')
weekly_cpi = cpi.resample('W-FRI').ffill()
weekly_cpi = weekly_cpi.rename(columns={'cpiaucscl': 'CPI'})

# Load Brent Oil (daily, resample weekly)
oil = pd.read_csv('data/DCOILBRENTEU.csv')
oil['observation_date'] = pd.to_datetime(oil['observation_date'])
oil = oil.set_index('observation_date')
weekly_oil = oil.resample('W-FRI').last()
weekly_oil = weekly_oil.rename(columns={weekly_oil.columns[-1]: 'Brent_Oil'})

# Load SP500 (daily, resample weekly)
sp500 = pd.read_csv('data/SP500.csv')
sp500['observation_date'] = pd.to_datetime(sp500['observation_date'])
sp500 = sp500.set_index('observation_date')
weekly_sp500 = sp500.resample('W-FRI').last()
weekly_sp500 = weekly_sp500.rename(columns={weekly_sp500.columns[-1]: 'SP500'})

# Load USD Index (daily, resample weekly)
usd = pd.read_csv('data/DTWEXBGS.csv')
usd['observation_date'] = pd.to_datetime(usd['observation_date'])
usd = usd.set_index('observation_date')
weekly_usd = usd.resample('W-FRI').last()
weekly_usd = weekly_usd.rename(columns={weekly_usd.columns[-1]: 'USD_Index'})

# Load Gold Price data (monthly, to be forward-filled weekly)
gold = pd.read_excel(
    'data/Gold_price_averages_in_a range_of_currencies_since_1978.xlsx',
    sheet_name='Monthly_Avg',
    header=5
)
gold = gold.drop(columns=['Unnamed: 0', 'Unnamed: 1'])
gold = gold.rename(columns={'Unnamed: 2': 'Date'})
gold = gold.dropna(subset=['Date', 'USD'])
gold['Date'] = pd.to_datetime(gold['Date'])
gold = gold.set_index('Date')
gold = gold.rename(columns={'USD': 'Gold_Price'})

# Resample gold data to weekly (Fridays) using forward fill
weekly_gold = gold.resample('W-FRI').ffill()

# Merge all on the date index (outer join to keep all available data)
df_final = weekly_cpi.join([weekly_oil, weekly_usd, weekly_sp500, weekly_gold], how='outer')

# Optionally, reset index and save to CSV
df_final = df_final.sort_index()
df_final = df_final.ffill()

df_final = df_final.reset_index()

df_final.to_csv('data/weekly_merged_data.csv', index=False)

print('Weekly merged data with gold prices saved to data/weekly_merged_data.csv') 