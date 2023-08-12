import numpy as np
import pandas as pd
from strategy1 import run_strategy_1
from helper.dc import calculate_dc

THETA_THRESHOLDS = (
    np.array(
        [0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]
    )
    / 100
)

df = pd.read_csv("data/stock_data.csv")


ALL_STOCK = df["ALL"]
uptrend, downtrend, p_ext = calculate_dc(ALL_STOCK, 0.098 / 100)

THETA_WEIGHTS = np.array(
    [0.1, 0.2, 0.05, 0.15, 0.1, 0.05, 0.1, 0.08, 0.07, 0.1]
)

r_df = run_strategy_1(THETA_THRESHOLDS, THETA_WEIGHTS, ALL_STOCK)
