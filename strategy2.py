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
from helper.dc import calculate_dc_indicators


BUY_COST_MULTIPLIER = 1.00025


def get_thresholds_decision(
    threshold_overshoot_summary: pd.DataFrame, prices: List[float]
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

    i = 0

    while i < len(threshold_overshoot_summary.index):
        if threshold_overshoot_summary.iloc[i]["event"] == "DR" and abs(
            threshold_overshoot_summary.iloc[i]["osv_cur"]
        ) >= abs(threshold_overshoot_summary.iloc[i]["osv_best"]):
            decisions[i] = "b"
        elif threshold_overshoot_summary.iloc[i]["event"] == "UR" and abs(
            threshold_overshoot_summary.iloc[i]["osv_cur"]
        ) >= abs(threshold_overshoot_summary.iloc[i]["osv_best"]):
            decisions[i] = "s"
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

        threshold_dc_summaries = calculate_dc_indicators(
            df[col], theta_thresholds
        )

        for i in range(len(theta_thresholds)):
            decisions = get_thresholds_decision(
                threshold_dc_summaries[i], df[col]
            )
            stock_decision_by_thresholds[col][f"threshold_{i}"] = decisions
    return stock_decision_by_thresholds


def get_stock_returns(
    df: pd.DataFrame,
    weights: List[float],
    stock_data: pd.DataFrame,
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
                    (row["prices"]) - (buy_price * BUY_COST_MULTIPLIER)
                ) / (buy_price)
                buy_price = 0

        if buy_price != 0:
            returns[-1] = (
                row["prices"] - (buy_price * BUY_COST_MULTIPLIER)
            ) / (buy_price)

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
    RoR = sum(returns_only)

    volatility = np.std(returns_only)
    sharpe_ratio = (RoR - risk_free_rate) / volatility

    return RoR, volatility, sharpe_ratio


def load_strategy_2(
    df: pd.DataFrame,
    thresholds: list,
    pkl_filename="data/strategy2_data.pkl",
    excel_filename="output/strategy2_output.xlsx",
    export_excel: bool = False,
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

    stock_decision_by_thresholds = {}

    # Check if the file exists
    if os.path.exists(pkl_filename):
        # If the file exists, load it
        with open(pkl_filename, "rb") as file:
            stock_decision_by_thresholds = pickle.load(file)
    else:
        stock_decision_by_thresholds = set_decisions(df, thresholds)

        if export_excel:
            # Create a new Excel writer object
            # pylint: disable=abstract-class-instantiated
            with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
                for (
                    sheet_name,
                    stock_data,
                ) in stock_decision_by_thresholds.items():
                    stock_data.to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )

        # If the file doesn't exist, save the dictionary to the file
        with open(pkl_filename, "wb") as file:
            pickle.dump(stock_decision_by_thresholds, file)

    return stock_decision_by_thresholds


def strategy2_fitness_function(
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

    sharpe_ratios = [0] * (len(df.columns) - 1)
    RoRs = [0] * (len(df.columns) - 1)
    volatility_list = [0] * (len(df.columns) - 1)

    stock_returns = get_stock_returns(df, weights, stock_data)

    for idx, col in enumerate(df.columns[1:]):
        RoR, volatility, sharpe_ratio = calculate_metrics(
            stock_returns[col], 0.025
        )

        sharpe_ratios[idx] = sharpe_ratio
        RoRs[idx] = RoR
        volatility_list[idx] = volatility

    return np.mean(RoRs), np.mean(volatility_list), np.mean(sharpe_ratios)
