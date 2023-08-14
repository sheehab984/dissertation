"""
This scripts runs the strategy1 based on thresholds, weights, and prices.
"""
from typing import List, Tuple, Union
from collections import namedtuple
import numpy as np
import pandas as pd
import datetime
import pygad

from strategy1_v1 import compute_threshold_dc_summaries


DCEvent = namedtuple("DCEvent", ["index", "price", "event"])
ThresholdSummary = namedtuple("ThresholdSummary", ["dc", "p_ext"])

# Read the data from CSV
df = pd.read_csv("data/stock_data.csv")

THETA_THRESHOLDS = (
    np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
)

BUY_COST_MULTIPLIER = 1.0025
THRESHOLDS_STOCK_DECISIONS = {col: pd.DataFrame() for col in df.columns[1:]}


def get_thresholds_decision(
    threshold_dc_summary: ThresholdSummary,
    prices: List[float],
) -> pd.DataFrame:
    decisions = []

    dc = threshold_dc_summary.dc
    p_ext = threshold_dc_summary.p_ext

    last_dc = pd.Series()
    time_interval_to_dc = 0
    last_decision = "h"
    new_decision = "h"

    for i in range(len(prices)):
        # Check if the index matches with the DC index
        if i in dc.index:
            last_price_ext = p_ext[p_ext.index < i].last_valid_index()
            time_interval_to_dc = i - last_price_ext
            last_dc = dc.loc[i]
            new_decision = "h"
        else:
            if not last_dc.empty:
                if time_interval_to_dc == (i - last_dc.name):
                    if last_dc["event"] == "DR" and (
                        last_decision == "s" or last_decision == "h"
                    ):
                        last_decision = "b"
                        new_decision = "b"
                    elif last_dc["event"] == "UR" and last_decision == "b":
                        last_decision = "s"
                        new_decision = "s"
                else:
                    new_decision = "h"
        decisions.append(new_decision)

    return decisions


def calculate_decision(row, weights):
    decisions_options = [("s", 0), ("h", 0), ("b", 0)]
    for i in range(1, len(weights) + 1):
        if row[i] == "b":
            decisions_options[2] = (
                "b",
                decisions_options[2][1] + weights[i - 1],
            )
        elif row[i] == "s":
            decisions_options[0] = (
                "s",
                decisions_options[0][1] + weights[i - 1],
            )
        else:
            decisions_options[1] = (
                "h",
                decisions_options[1][1] + weights[i - 1],
            )

    return max(decisions_options, key=lambda x: x[1])[0]


def set_decisions():
    for col in df.columns[1:]:
        THRESHOLDS_STOCK_DECISIONS[col] = pd.DataFrame(
            {
                "prices": df[col],
            }
        )
        threshold_dc_summaries = compute_threshold_dc_summaries(
            df[col], THETA_THRESHOLDS
        )

        for i in range(len(THETA_THRESHOLDS)):
            decisions = get_thresholds_decision(
                threshold_dc_summaries[i], df[col]
            )
            THRESHOLDS_STOCK_DECISIONS[col][f"threshold_{i}"] = decisions


def run_fitness_function(weights, run_id=1):
    for col in df.columns[1:]:
        stock_df = THRESHOLDS_STOCK_DECISIONS[col]
        last_buy_price = 0
        returns = [0] * stock_df.shape[0]
        last_decision = "h"

        for i in range(stock_df.shape[0]):
            row = stock_df.loc[i]
            new_decision = calculate_decision(row, weights)
            if new_decision == "b" and (
                last_decision == "s" or last_decision == "h"
            ):
                last_decision = new_decision
                last_buy_price = row["prices"] * BUY_COST_MULTIPLIER
            elif new_decision == "s" and last_decision == "b":
                last_decision = new_decision
                returns[i] = row["prices"] - last_buy_price
        THRESHOLDS_STOCK_DECISIONS[col][f"returns_{run_id}"] = returns


def calculate_metrics(returns, risk_free_rate=0.01):
    """
    Calculate RoR, Risk, and Sharpe Ratio from a return array.

    Parameters:
    - returns (list or np.array): Array of returns.
    - risk_free_rate (float): Risk-free rate. Default is 0.01 (1%).

    Returns:
    - RoR, Risk, Sharpe Ratio
    """

    # Convert returns to numpy array for easier calculations
    returns = np.array(returns)

    # Calculate RoR
    RoR = np.mean(returns)

    # Calculate Risk (Volatility)
    Risk = np.std(returns)

    # Calculate Sharpe Ratio
    Sharpe_Ratio = (RoR - risk_free_rate) / Risk

    return RoR, Risk, Sharpe_Ratio


def compute_sharpe_ratio(weights, solution_idx):
    returns = [0] * df.columns[1:].shape[0]
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{current_time}: This will be appended to the file with a timestamp."
    )
    print("-------------------------------------------------------")
    print(f"New Weights are: {weights}")

    run_fitness_function(weights, solution_idx)

    for idx, col in enumerate(df.columns[1:]):
        stock_df = THRESHOLDS_STOCK_DECISIONS[col]
        stock_returns = stock_df[f"returns_{solution_idx}"]
        _, _, sharpe_Ratio = calculate_metrics(stock_returns, 0.025)
        print(f"Sharpe Ratio for {col} is: {sharpe_Ratio}")
        returns[idx] = sharpe_Ratio

    print(f"Fitness is: {np.mean(returns)}")
    return np.mean(returns)


# The fitness function
def fitness_func(ga_instance, solution, solution_idx):
    # The objective is to maximize the sum of squares
    return compute_sharpe_ratio(solution, solution_idx)


if __name__ == "__main__":
    set_decisions()

    # Create an instance of the GA class
    ga_instance = pygad.GA(
        num_generations=18,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=10,
        gene_type=np.float32,
        init_range_low=0,
        init_range_high=1,
        parent_selection_type="tournament",
        K_tournament=2,
        crossover_type="single_point",
        crossover_probability=0.95,
        mutation_type="random",
        mutation_probability=0.05,
    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution is:", solution)
    print("Fitness of the best solution is:", solution_fitness)

    with pd.ExcelWriter("output.xlsx", engine="openpyxl") as writer:
        for sheet_name, df in THRESHOLDS_STOCK_DECISIONS.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
