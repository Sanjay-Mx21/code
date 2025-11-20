import pandas as pd

def compute_returns(price_df):
    """Simple log returns (daily)"""
    return price_df.pct_change().dropna()
