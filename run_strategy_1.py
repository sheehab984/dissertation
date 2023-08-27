import numpy as np
import pandas as pd
import logging

from strategy1 import load_strategy_1, strategy1_fitness_function

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app_test_1.log",
    filemode="w",
)  # 'w' will overwrite the log file each time the script runs. Use 'a' to append.

# Create a logger object
logger = logging.getLogger()


def split_func(df):
    # Define the split ratios
    train_ratio = 0.8

    # Calculate the split indices
    total_rows = len(df)
    train_split_idx = int(total_rows * train_ratio)

    # Split the data
    train_df = df.iloc[:train_split_idx].reset_index(drop=True)
    test_df = df.iloc[train_split_idx:].reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    weights = [
        [
            0.13037498,
            0.04352858,
            0.00368317,
            0.00465341,
            0.17115107,
            0.01095872,
            0.01412538,
            0.01354227,
            0.10388808,
            0.50409436,
        ],
        [
            1.0648172e-03,
            4.2600158e-04,
            1.6502303e-01,
            1.5927916e-03,
            2.3416087e-03,
            1.0402963e-01,
            9.3359742e-03,
            1.9533007e-01,
            4.2510568e-03,
            5.1660502e-01,
        ],
        [
            0.0370706,
            0.0218132,
            0.02519661,
            0.025277,
            0.02614085,
            0.02185163,
            0.01410312,
            0.20015036,
            0.11767706,
            0.51071954,
        ],
    ]

    # Read the data from CSV
    df = pd.read_csv("data/stock_data.csv")

    train_df, test_df = split_func(df)

    # Define thresholds
    thresholds = (
        np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55])
        / 100
    )

    # Load strategy 2 decisions
    stock_decision_by_thresholds = load_strategy_1(
        test_df, thresholds, "data/strategy1_test_data.pkl"
    )

    for weight in weights:
        # Use the solution to generate trading signals and calculate returns for the training set
        RoR, volatility, sharpe_ratio = strategy1_fitness_function(
            test_df, weight, stock_decision_by_thresholds
        )

        print(
            "Weights: "
            + str(weight)
            + "\n"
            + "Test fitness function for solution "
            + str(sharpe_ratio)
            + "\t"
            + str(volatility)
            + "\t"
            + str(RoR)
        )
        logger.debug(
            "Weights: "
            + str(weight)
            + "\n"
            + "Test fitness function for solution "
            + str(sharpe_ratio)
            + "\t"
            + str(volatility)
            + "\t"
            + str(RoR)
        )
