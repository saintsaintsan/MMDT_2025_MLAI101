import pandas as pd
from pandas.tseries.offsets import MonthEnd

# Read the monthly CPIAUCSL data
input_path = 'data/CPIAUCSL.csv'
df = pd.read_csv(input_path)
print('Original monthly data:')
print(df.head())

# Convert observation_date to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

daily_rows = []
for _, row in df.iterrows():
    start_date = row['observation_date']
    end_date = (start_date + MonthEnd(0))
    days = pd.date_range(start=start_date, end=end_date, freq='D')
    for day in days:
        daily_rows.append({'date': day.strftime('%Y-%m-%d'), 'CPIAUCSL': row['CPIAUCSL']})

daily_df = pd.DataFrame(daily_rows)
output_path = 'data/CPIAUCSL_daily.csv'
daily_df.to_csv(output_path, index=False)
print(f'Expanded daily data saved to {output_path}')
print(daily_df.tail(35))  # Show first month as a sample
