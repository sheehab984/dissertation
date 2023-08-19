"""
This script runs the strategy1 based on thresholds, weights, and prices.
"""
import datetime
import os
import pickle
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from helper.dc import (
    calculate_dc_indicators,
    compute_threshold_dc_summaries,
    ThresholdSummary,
)


BUY_COST_MULTIPLIER = 1.0025
SELL_COST_MULTIPLIER = 0.9975


def get_thresholds_decision(
    threshold_overshoot_summary: pd.DataFrame,
    threshold_dc_summary: ThresholdSummary,
    prices: List[float],
) -> List[str]:
    """
    Get decisions based on threshold summaries and prices.

    Parameters:
    - threshold_dc_summary (ThresholdSummary2): Summary of the threshold.
    - prices (List[float]): List of prices.

    Returns:
    - List[str]: Decisions for each price.
    """
    decisions = ["h"] * len(prices)
    dc = threshold_dc_summary.dc
    i = 0
    buy_counter = 0
    bought_index = -1

    while i < len(prices):
        if bought_index != -1:
            dc_index = dc[
                (dc.index > bought_index) & (dc.event == "DR")
            ].first_valid_index()
            if dc_index is not None:
                decisions[dc_index] = "s"
                bought_index = -1
                while i != dc_index:
                    i += 1
        if i in threshold_overshoot_summary.index:
            if threshold_overshoot_summary.loc[i]["event"] == "UR":
                buy_counter += 1
                if buy_counter == 3:
                    decisions[i] = "b"
                    bought_index = i
            elif threshold_overshoot_summary.loc[i]["event"] == "DR":
                buy_counter = 0
        i += 1

    return decisions


def calculate_decision(row: pd.Series, weights: List[float]) -> str:
    """
    Calculate decision based on row data and weights.

    Parameters:
    - row (pd.Series): Row data.
    - weights (List[float]): Weights for decision.

    Returns:
    - str: Decision.
    """
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


def set_decisions(
    df: pd.DataFrame, theta_thresholds: List[float]
) -> Dict[str, pd.DataFrame]:
    """
    Set decisions based on dataframe and thresholds.

    Parameters:
    - df (pd.DataFrame): Dataframe with data.
    - theta_thresholds (List[float]): List of thresholds.

    Returns:
    - Dict[str, pd.DataFrame]: Decision by thresholds.
    """
    stock_decision_by_thresholds = {}

    for col in df.columns[1:]:
        stock_decision_by_thresholds[col] = pd.DataFrame({"prices": df[col]})

        threshold_overshoot_summaries = calculate_dc_indicators(
            df[col], theta_thresholds
        )

        threshold_dc_summaries = compute_threshold_dc_summaries(
            df[col], theta_thresholds
        )

        for i in range(len(theta_thresholds)):
            decisions = get_thresholds_decision(
                threshold_overshoot_summaries[i],
                threshold_dc_summaries[i],
                df[col],
            )
            stock_decision_by_thresholds[col][f"threshold_{i}"] = decisions
    return stock_decision_by_thresholds


def get_stock_returns(
    df: pd.DataFrame, weights: List[float], stock_data: pd.DataFrame
) -> Dict[str, List[Union[float, None]]]:
    """
    Get stock returns based on dataframe, weights, and stock data.

    Parameters:
    - df (pd.DataFrame): Dataframe with data.
    - weights (List[float]): Weights for decision.
    - stock_data (pd.DataFrame): Stock data.

    Returns:
    - Dict[str, List[Union[float, None]]]: Stock returns.
    """
    stock_returns = {}
    for col in df.columns[1:]:
        stock_df = stock_data[col]
        returns = [None] * stock_df.shape[0]
        buy_price = 0

        last_decision = "h"

        for i in range(stock_df.shape[0]):
            row = stock_df.loc[i]
            new_decision = calculate_decision(row, weights)
            if new_decision == "b" and (
                last_decision == "s" or last_decision == "h"
            ):
                last_decision = new_decision
                buy_price = row["prices"]
            elif new_decision == "s" and last_decision == "b":
                last_decision = new_decision
                returns[i] = (
                    (row["prices"] * SELL_COST_MULTIPLIER)
                    - (buy_price * BUY_COST_MULTIPLIER)
                ) / (buy_price * BUY_COST_MULTIPLIER)
        stock_returns[col] = returns
    return stock_returns


def calculate_metrics(
    returns: List[Union[float, None]], risk_free_rate: float = 0.01
) -> Tuple[float, float, float]:
    """
    Calculate RoR, Risk, and Sharpe Ratio from a return array.

    Parameters:
    - returns (List[Union[float, None]]): Array of returns.
    - risk_free_rate (float): Risk-free rate. Default is 0.01 (1%).

    Returns:
    - Tuple[float, float, float]: RoR, Risk, Sharpe Ratio.
    """
    returns = np.array(returns)
    returns_only = returns[returns != np.array(None)]
    RoR = returns_only.mean()

    volatility = np.std(returns_only)
    sharpe_ratio = (RoR - risk_free_rate) / volatility

    return RoR, volatility, sharpe_ratio


def load_strategy_3(
    df: pd.DataFrame, thresholds: list, export_excel: bool = False
) -> dict:
    """
    Load strategy 1 data. If the data file exists, it reads from the file.
    Otherwise, it sets decisions based on the provided dataframe and thresholds,
    and optionally exports the results to an Excel file.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - thresholds (list): List of thresholds for making decisions.
    - export_excel (bool, optional): Whether to export the results to an Excel file. Defaults to False.

    Returns:
    - dict: Dictionary containing decisions by thresholds.
    """

    filename = "data/strategy3_data.pkl"
    stock_decision_by_thresholds = {}

    # Check if the file exists
    if os.path.exists(filename):
        # If the file exists, load it
        with open(filename, "rb") as file:
            stock_decision_by_thresholds = pickle.load(file)
    else:
        stock_decision_by_thresholds = set_decisions(df, thresholds)

        if export_excel:
            # Create a new Excel writer object
            # pylint: disable=abstract-class-instantiated
            with pd.ExcelWriter(
                "output/strategy3_output.xlsx", engine="openpyxl"
            ) as writer:
                for (
                    sheet_name,
                    stock_data,
                ) in stock_decision_by_thresholds.items():
                    stock_data.to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )

        # If the file doesn't exist, save the dictionary to the file
        with open(filename, "wb") as file:
            pickle.dump(stock_decision_by_thresholds, file)

    return stock_decision_by_thresholds


def strategy3_fitness_function(
    df: pd.DataFrame, weights: list, stock_data: pd.DataFrame
) -> float:
    """
    Calculate the fitness of strategy 1 based on Sharpe Ratios for given weights.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - weights (list): List of weights for making decisions.
    - stock_data (pd.DataFrame): Dataframe containing stock data.

    Returns:
    - float: Mean of the Sharpe Ratios, representing the fitness of the strategy.
    """

    returns = [0] * (len(df.columns) - 1)

    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{current_time}: This will be appended to the file with a timestamp."
    )
    print("-------------------------------------------------------")
    print(f"New Weights are: {weights}")

    stock_returns = get_stock_returns(df, weights, stock_data)

    for idx, col in enumerate(df.columns[1:]):
        _, _, sharpe_ratio = calculate_metrics(stock_returns[col], 0.025)
        print(f"Sharpe Ratio for {col} is: {sharpe_ratio}")
        returns[idx] = sharpe_ratio

    print(f"Fitness is: {np.mean(returns)}")
    return np.mean(returns)
