
import yfinance as yf
import pandas as pd

import numpy as np

# List of tickers
tickers = ["ALL", "ASGN", "CI", "COP", "EME", "EVR", "GILD", "GPK", 
           "ISRG", "MKL", "MOH", "PEG", "PXD", "QCOM", "UBSI", "VFC", "XEL"]

# Define the start and end dates
start_date = "2009-11-27"
end_date = "2019-11-27"

# Fetch data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Export to CSV
data.to_csv('stock_data.csv')

print("Data exported to stock_data.csv")




# -*- coding: utf-8 -*-

