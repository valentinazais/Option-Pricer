import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# Paths to data 
spot_path = "///.csv"
iv_path = "///.csv"
dividend_path = "///.csv"
rfr_path = "///.csv"

# Read CSVs
spot_df = pd.read_csv(spot_path, index_col=0, parse_dates=True, sep=";")
iv_df = pd.read_csv(iv_path, index_col=0, parse_dates=True, sep=";")
dividend_df = pd.read_csv(dividend_path, index_col=0, parse_dates=True, sep=";")
rfr_df = pd.read_csv(rfr_path, index_col=0, parse_dates=True, sep=";")

# Optional
for df in [spot_df, iv_df, dividend_df, rfr_df]:
    df.replace(',', '.', regex=True, inplace=True)
    df = df.astype(float, errors='ignore')

# Optional
spot_df.index = pd.to_datetime(spot_df.index, format='%m/%d/%Y', errors='coerce')
rfr_df.index = pd.to_datetime(rfr_df.index, format='%m/%d/%Y', errors='coerce')


iv_df.index = pd.to_datetime(iv_df.index, format='%d/%m/%Y', errors='coerce')
dividend_df.index = pd.to_datetime(dividend_df.index, format='%d/%m/%Y', errors='coerce')

# Clean the data
spot_df = spot_df[spot_df.index.notna()]
rfr_df = rfr_df[rfr_df.index.notna()]
iv_df = iv_df[iv_df.index.notna()]
dividend_df = dividend_df[dividend_df.index.notna()]

# Date filter (to change as input)
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2025-12-31')  # Example end date, adjust as needed
spot_df = spot_df[(spot_df.index >= start_date) & (spot_df.index <= end_date)]
rfr_df = rfr_df[(rfr_df.index >= start_date) & (rfr_df.index <= end_date)]
iv_df = iv_df[(iv_df.index >= start_date) & (iv_df.index <= end_date)]
dividend_df = dividend_df[(dividend_df.index >= start_date) & (dividend_df.index <= end_date)]

# Black-Scholes formula for put option price
def black_scholes_put(S, K, T, r, q, sigma):
    """
    S: Spot price
    K: Strike price (95% moneyness, so K = S * 0.95 for puts)
    T: Time to maturity in years (30 days = 30/365)
    r: Risk-free rate (continuous)
    q: Dividend yield (continuous)
    sigma: Implied volatility
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

# Time to maturity 
T = 30 / 365.0

# Moneyness (100% = ATM)
M = 87.5%

# Create a new df for put prices
tickers = spot_df.columns
put_price_df = pd.DataFrame(index=spot_df.index, columns=tickers)

common_dates = spot_df.index.intersection(iv_df.index).intersection(dividend_df.index).intersection(rfr_df.index)
put_price_df = put_price_df.loc[common_dates]

# Compute put prices for each ticker and each day
for date in common_dates:
    for ticker in tickers:
        # Get spot price
        S = spot_df.loc[date, ticker]
        # Get implied volatility
        sigma = iv_df.loc[date, ticker] if ticker in iv_df.columns else np.nan
        # Convert sigma to float if necessary
        try:
            sigma = float(sigma) if not pd.isna(sigma) else np.nan
        except (ValueError, TypeError):
            sigma = np.nan
        # Get dividend yield (replace NaN with 0)
        q = dividend_df.loc[date, ticker] if ticker in dividend_df.columns else 0
        # Convert q to float if necessary
        try:
            q = float(q) if not pd.isna(q) else 0
        except (ValueError, TypeError):
            q = 0
        # Get risk-free rate (same for all tickers, column name 'USGG1M_Index')
        r = rfr_df.loc[date, 'USGG1M_Index']
        # Convert r to float if necessary
        try:
            r = float(r) if not pd.isna(r) else np.nan
        except (ValueError, TypeError):
            r = np.nan
        
        # Check for NaN in spot price or implied volatility
        if pd.isna(S) or pd.isna(sigma) or pd.isna(r):
            put_price_df.loc[date, ticker] = np.nan
        else:
             
            K = S * M
            
            r_continuous = np.log(1 + r / 100) if r > 0 else r / 100
            q_continuous = np.log(1 + q / 100) if q > 0 else q / 100
            sigma_decimal = sigma / 100 
            
            put_price = black_scholes_put(S, K, T, r_continuous, q_continuous, sigma_decimal)
            put_price_df.loc[date, ticker] = put_price

# Save df
output_dir = "///Output"
output_path = os.path.join(output_dir, "NDX_OTM_Put30D_Prices_875Moneyness_2020_2025.csv")

# Optional
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Save the df to csv
put_price_df.to_csv(output_path, sep=";", decimal=".", index=True)
print(f"DataFrame saved to {output_path}")
