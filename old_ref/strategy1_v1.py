"""
This scripts runs the strategy1 based on thresholds, weights, and prices.
"""
from typing import List, Tuple, Union
from collections import namedtuple
import numpy as np
import pandas as pd

from helper.dc import calculate_dc


DCEvent = namedtuple("DCEvent", ["index", "price", "event"])
ThresholdSummary = namedtuple("ThresholdSummary", ["dc", "p_ext"])

BUY_COST_MULTIPLIER = 1.0025


def calculate_decision(
    thresholds: List[float],
    thresholds_last_decision: List[str],
    weights: List[float],
) -> str:
    """
    Calculate decision based on thresholds and weights.

    Args:
    - thresholds: List of threshold values.
    - thresholds_last_decision: List of last decisions for each threshold.
    - weights: List of weights for each threshold.

    Returns:
    - Decision string ('s', 'h', or 'b').
    """
    decisions = {"s": 0, "h": 0, "b": 0}
    no_decision_count = 0
    for j, threshold in enumerate(thresholds):
        decision = thresholds_last_decision[j]
        if decision is not None:
            decisions[decision] += weights[j]
        else:
            no_decision_count += 1
    if no_decision_count == len(thresholds):
        return "h"

    return max(decisions, key=decisions.get)


def run_strategy_1(
    threshold_dc_summaries: pd.DataFrame,
    thresholds: List[float],
    weights: List[float],
    prices: List[float],
) -> pd.DataFrame:
    """
    Run strategy based on thresholds, weights, and prices.

    Args:
    - thresholds: List of threshold values.
    - weights: List of weights for each threshold.
    - prices: List of price values.

    Returns:
    - DataFrame containing return prices.
    """

    thresholds_last_dc = [None] * len(thresholds)
    thresholds_last_dc_time_interval = [-1] * len(thresholds)
    thresholds_last_decision = [None] * len(thresholds)
    last_position = None
    last_buy_price = 0
    return_prices = pd.DataFrame({"prices": prices, "returns": 0})
    for i, price in enumerate(prices):
        (
            thresholds_last_dc,
            thresholds_last_dc_time_interval,
            thresholds_last_decision,
        ) = update_thresholds_last_dc(
            i,
            thresholds,
            threshold_dc_summaries,
            thresholds_last_dc,
            thresholds_last_dc_time_interval,
            thresholds_last_decision,
        )
        new_position = calculate_decision(
            thresholds, thresholds_last_decision, weights
        )
        (
            last_buy_price,
            last_position,
            return_prices,
        ) = execute_strategy(
            i,
            price,
            new_position,
            last_position,
            thresholds_last_dc,
            last_buy_price,
            return_prices,
            thresholds,
        )

    return return_prices


def compute_threshold_dc_summaries(
    prices: List[float], thresholds: List[float]
) -> List[ThresholdSummary]:
    """
    Compute summaries for each threshold based on price data.

    Args:
    - prices: List of price values.
    - thresholds: List of threshold values.

    Returns:
    - List of ThresholdSummary objects containing DC data and p_ext data.
    """
    summaries = []
    for threshold in thresholds:
        upturn, downturn, p_ext = calculate_dc(prices, threshold)
        upturn = [DCEvent(x[0], x[1], "UR") for x in upturn]
        downturn = [DCEvent(x[0], x[1], "DR") for x in downturn]
        p_ext = [DCEvent(x[1], x[0], x[2]) for x in p_ext]
        dc_data, p_ext_data = merge_dc_events(upturn, downturn, p_ext)
        summaries.append(ThresholdSummary(dc_data, p_ext_data))
    return summaries


def merge_dc_events(
    upturn: List[DCEvent],
    downturn: List[DCEvent],
    p_ext: List[DCEvent],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge upturn and downturn events into a single DataFrame.

    Args:
    - upturn: List of upturn DC events.
    - downturn: List of downturn DC events.
    - p_ext: List of p_ext events.

    Returns:
    - Tuple of DataFrames containing merged DC data and p_ext data.
    """
    dc_indexes, dc_prices, dc_event = [], [], []
    if len(upturn) > len(downturn):
        events = list(zip(upturn, downturn))
    else:
        events = list(zip(downturn, upturn))
    for a, b in events:
        dc_indexes.extend([a.index, b.index])
        dc_prices.extend([a.price, b.price])
        dc_event.extend([a.event, b.event])
    dc_data = pd.DataFrame(
        data={"price": np.array(dc_prices), "event": dc_event},
        index=dc_indexes,
    )
    p_ext_data = pd.DataFrame(
        data={
            "price": [x.price for x in p_ext],
            "event": [x.event for x in p_ext],
        },
        index=[x.index for x in p_ext],
    )
    return dc_data, p_ext_data


def update_thresholds_last_dc(
    i: int,
    thresholds: List[float],
    threshold_dc_summaries: List[ThresholdSummary],
    thresholds_last_dc: List[Union[None, pd.Series]],
    thresholds_last_dc_time_interval: List[int],
    thresholds_last_decision: List[Union[None, str]],
) -> None:
    """
    Update the last DC values for each threshold based on the current price index.

    Args:
    - i: Current price index.
    - thresholds: List of threshold values.
    - threshold_dc_summaries: List of ThresholdSummary objects.
    - thresholds_last_dc: List of last DC values for each threshold.
    - thresholds_last_dc_time_interval: List of time intervals
    since the last DC event for each threshold.
    - thresholds_last_decision: List of last decisions for each threshold.

    Returns:
    - updated thresholds_last_dc: List of last DC values for each threshold.
    - updated thresholds_last_dc_time_interval: List of time intervals
    since the last DC event for each threshold.
    - updated thresholds_last_decision: List of last decisions for each threshold.
    """
    for j, threshold in enumerate(thresholds):
        dc = threshold_dc_summaries[j].dc
        p_ext = threshold_dc_summaries[j].p_ext
        if i in dc.index:
            last_price_ext = p_ext[p_ext.index < i].last_valid_index()
            time_interval_to_dc = i - last_price_ext
            thresholds_last_dc[j] = dc.loc[i]
            thresholds_last_dc_time_interval[j] = time_interval_to_dc
        else:
            thresholds_last_decision = update_decision(
                i,
                thresholds_last_dc[j],
                thresholds_last_dc_time_interval[j],
                thresholds_last_decision,
                j,
            )

    return (
        thresholds_last_dc,
        thresholds_last_dc_time_interval,
        thresholds_last_decision,
    )


def update_decision(
    i: int,
    last_dc: Union[None, pd.Series],
    last_dc_time_interval: int,
    thresholds_last_decision: List[Union[None, str]],
    j: int,
) -> None:
    """
    Update the decision for the current threshold based on the last DC event.

    Args:
    - i: Current price index.
    - last_dc: Last DC value for the current threshold.
    - last_dc_time_interval: Time interval since the last DC event for the current threshold.
    - thresholds_last_decision: List of last decisions for each threshold.
    - j: Index of the current threshold.

    Returns:
    - updated thresholds_last_decision: List of last decisions for each threshold.
    """
    if not isinstance(last_dc, pd.Series):
        return thresholds_last_decision
    if last_dc_time_interval == (i - last_dc.name):
        if last_dc["event"] == "DR":
            thresholds_last_decision[j] = "b"
        elif last_dc["event"] == "UR":
            thresholds_last_decision[j] = "s"
    else:
        thresholds_last_decision[j] = "h"
    return thresholds_last_decision


def execute_strategy(
    i: int,
    price: float,
    new_position: str,
    last_position: str,
    thresholds_last_dc: List[Union[None, pd.Series]],
    last_buy_price: float,
    return_prices: pd.DataFrame,
    thresholds: List[float],
) -> None:
    """
    Execute the trading strategy based on the current position and price.

    Args:
    - i: Current price index.
    - price: Current price value.
    - new_position: New position to take ('b', 's', or 'h').
    - last_position: Last position taken.
    - thresholds_last_dc: List of last DC values for each threshold.
    - last_buy_price: Last price at which a buy was executed.
    - return_prices: DataFrame containing return prices.
    - thresholds: List of threshold values.

    Returns:
    - last_buy_price: Last price at which a buy was executed.
    - updated last_position: Last position taken.
    - return_prices: DataFrame containing return prices.
    """
    if new_position != "h":
        dr_count = sum(
            1
            for td in thresholds_last_dc
            if isinstance(td, pd.Series) and td["event"] == "DR"
        )
        ur_count = len(thresholds) - dr_count
        if new_position == "b" and last_position != "b" and dr_count > ur_count:
            last_buy_price = price * BUY_COST_MULTIPLIER
        elif (
            new_position == "s" and last_position != "s" and dr_count < ur_count
        ):
            return_prices.loc[i, "returns"] = last_buy_price - price

        last_position = new_position

    return last_buy_price, last_position, return_prices
