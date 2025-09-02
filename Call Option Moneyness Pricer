import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# Paths to data
spot_path = "///.csv"
iv_path = "///.csv"
dividend_path = "///.csv"
rfr_path = "///.csv"

# Reading CSVs
spot_df = pd.read_csv(spot_path, index_col=0, parse_dates=True, sep=";")
iv_df = pd.read_csv(iv_path, index_col=0, parse_dates=True, sep=";")
dividend_df = pd.read_csv(dividend_path, index_col=0, parse_dates=True, sep=";")
rfr_df = pd.read_csv(rfr_path, index_col=0, parse_dates=True, sep=";")

#Optional
for df in [spot_df, iv_df, dividend_df, rfr_df]:
    df.replace(',', '.', regex=True, inplace=True)
    df = df.astype(float, errors='ignore')

# Optional
spot_df.index = pd.to_datetime(spot_df.index, format='%m/%d/%Y', errors='coerce')
rfr_df.index = pd.to_datetime(rfr_df.index, format='%m/%d/%Y', errors='coerce')


iv_df.index = pd.to_datetime(iv_df.index, format='%d/%m/%Y', errors='coerce')
dividend_df.index = pd.to_datetime(dividend_df.index, format='%d/%m/%Y', errors='coerce')

# Cleaning
spot_df = spot_df[spot_df.index.notna()]
rfr_df = rfr_df[rfr_df.index.notna()]
iv_df = iv_df[iv_df.index.notna()]
dividend_df = dividend_df[dividend_df.index.notna()]

# Filter date
start_date = pd.to_datetime('2020-01-01')
spot_df = spot_df[spot_df.index >= start_date]
rfr_df = rfr_df[rfr_df.index >= start_date]
iv_df = iv_df[iv_df.index >= start_date]
dividend_df = dividend_df[dividend_df.index >= start_date]

# Black-Scholes formula for call option price
def black_scholes_call(S, K, T, r, q, sigma):
    """
    S: Spot price
    K: Strike price (ATM, so K = S)
    T: Time to maturity in years (30 days = 30/365)
    r: Risk-free rate (continuous)
    q: Dividend yield (continuous)
    sigma: Implied volatility
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Time to maturity
T = 30 / 365.0

# Moneyness (100% = ATM)
M = 95%


tickers = spot_df.columns
call_price_df = pd.DataFrame(index=spot_df.index, columns=tickers)


common_dates = spot_df.index.intersection(iv_df.index).intersection(dividend_df.index).intersection(rfr_df.index)
call_price_df = call_price_df.loc[common_dates]


for date in common_dates:
    for ticker in tickers:
        
        S = spot_df.loc[date, ticker]
        
        sigma = iv_df.loc[date, ticker] if ticker in iv_df.columns else np.nan
        
        try:
            sigma = float(sigma) if not pd.isna(sigma) else np.nan
        except (ValueError, TypeError):
            sigma = np.nan
        
        q = dividend_df.loc[date, ticker] if ticker in dividend_df.columns else 0
        
        try:
            q = float(q) if not pd.isna(q) else 0
        except (ValueError, TypeError):
            q = 0
        
        r = rfr_df.loc[date, 'USGG1M_Index']
        
        try:
            r = float(r) if not pd.isna(r) else np.nan
        except (ValueError, TypeError):
            r = np.nan
        
        
        if pd.isna(S) or pd.isna(sigma) or pd.isna(r):
            call_price_df.loc[date, ticker] = np.nan
        else:
        
            K = S * M
            # Convert rates to continuous if they are in percentage
            r_continuous = np.log(1 + r / 100) if r > 0 else r / 100
            q_continuous = np.log(1 + q / 100) if q > 0 else q / 100
            sigma_decimal = sigma / 100  # Convert IV from percentage to decimal
            
        
            call_price = black_scholes_call(S, K, T, r_continuous, q_continuous, sigma_decimal)
            call_price_df.loc[date, ticker] = call_price

# Optional
print("ATM Call 30D Prices DataFrame (from 2020 onwards):")
print(call_price_df.head())

# Save the DataFrame to a CSV 
output_dir = ".../Output"
output_path = os.path.join(output_dir, "NDX_ATM_Call30D_Prices_2020.csv")

# Optional
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Save the DataFrame to CSV
call_price_df.to_csv(output_path, sep=";", decimal=".", index=True)
print(f"DataFrame saved to {output_path}")
