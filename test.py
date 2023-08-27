import numpy as np
import pandas as pd

from strategy1 import load_strategy_1, strategy1_fitness_function


def split_func(df):
    # Define the split ratios
    train_ratio = 0.8

    # Calculate the split indices
    total_rows = len(df)
    train_split_idx = int(total_rows * train_ratio)

    # Split the data
    train_df1 = df.iloc[:train_split_idx].reset_index(drop=True)
    test_df1 = df.iloc[train_split_idx + 3 :].reset_index(drop=True)

    return train_df1, test_df1


# Read the data from CSV
df = pd.read_csv("data/stock_data.csv")

train_df, test_df = split_func(df)
# Define thresholds
main_thresholds = (
    np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
)
for threshold in main_thresholds:
    # Load strategy 1 decisions
    stock_decision_by_threshold = load_strategy_1(
        df=test_df,
        thresholds=[threshold],
        pkl_filename=f"data/strategy1{threshold}_test_data.pkl",
    )

    RoR, volatility, sharpe_ratio = strategy1_fitness_function(
        test_df, [1], stock_decision_by_threshold
    )

    print(f"Strategy 1 matrices for threshold {threshold * 100}:")
    print(
        f"RoR: {RoR:.5f}, Risk: {volatility:.5f}, Sharpe Ratio: {sharpe_ratio:.5f}"
    )
