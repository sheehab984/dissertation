"""
This script contains functions for calculating Directional Change (DC) and
related indicators.
"""
from typing import List, Tuple
import numpy as np


# flake8: noqa: C901
def calculate_dc(
    prices: List[float], threshold: float
) -> Tuple[
    List[Tuple[int, float]],
    List[Tuple[int, float]],
    List[Tuple[float, int, str]],
]:
    """
    Calculate Directional Change (DC) based on given price data and threshold.

    Parameters:
    - prices: A list of price data.
    - threshold: A threshold value to determine upturns and downturns.

    Returns:
    - upturn, downturn, extreme price points
    """

    last_low_index = 0
    last_high_index = 0
    last_low_price = prices[last_low_index]
    last_high_price = prices[last_high_index]

    upturn_dc = []
    downturn_dc = []
    p_ext = []

    current_index = 1

    # First while loop: Determine
    # the initial event (either an upturn or downturn)
    # This loop will break once the first event is identified.
    while current_index < len(prices):
        if prices[current_index] <= last_high_price * (1 - threshold):
            downturn_dc.append((current_index, prices[current_index]))
            p_ext.append((last_high_price, last_high_index, "DR"))
            event = "DR"
            break
        elif prices[current_index] >= last_low_price * (
            1 + threshold
        ):
            upturn_dc.append((current_index, prices[current_index]))
            p_ext.append((last_low_price, last_low_index, "UR"))
            event = "UR"
            break
        elif prices[current_index] > last_high_price:
            last_high_index = current_index
            last_high_price = prices[current_index]
        elif prices[current_index] < last_low_price:
            last_low_index = current_index
            last_low_price = prices[current_index]
        current_index += 1

    # Second while loop: Determine subsequent events based on the initial event
    # This loop will continue until all prices are processed.
    while current_index < len(prices):
        if event == "DR":
            if prices[current_index] < last_low_price:
                last_low_index = current_index
                last_low_price = prices[current_index]
            elif prices[current_index] >= last_low_price * (
                1 + threshold
            ):
                upturn_dc.append(
                    (current_index, prices[current_index])
                )
                p_ext.append((last_low_price, last_low_index, "UR"))
                last_high_index = current_index
                last_high_price = prices[current_index]
                event = "UR"
        elif event == "UR":
            if prices[current_index] > last_high_price:
                last_high_index = current_index
                last_high_price = prices[current_index]
            elif prices[current_index] <= last_high_price * (
                1 - threshold
            ):
                downturn_dc.append(
                    (current_index, prices[current_index])
                )
                p_ext.append((last_high_price, last_high_index, "DR"))
                event = "DR"
                last_low_index = current_index
                last_low_price = prices[current_index]
        current_index += 1

    return upturn_dc, downturn_dc, p_ext


def calculate_dc_indicators(
    prices: List[float],
    upturn: List[Tuple[int, float]],
    downturn: List[Tuple[int, float]],
    p_ext: List[Tuple[float, int, str]],
    threshold: float,
    chunk_size: int = 4,
) -> List[Tuple[int, float, float, str, float]]:
    """
    Calculate DC indicators based on given parameters.

    Args:
    - prices: List of price values.
    - upturn: List of upturn events.
    - downturn: List of downturn events.
    - p_ext: List of price extension events.
    - threshold: Threshold value for calculation.
    - chunk_size: Size of chunks for splitting overshoot data.

    Returns:
    - List of tuples containing overshoot data with best OSV.
    """
    all_overshoot = compute_all_overshoot(
        prices, upturn, downturn, p_ext, threshold
    )
    chunks = split_into_chunks(all_overshoot, chunk_size)
    medians = [np.median([x[2] for x in chunk]) for chunk in chunks]
    all_overshoot_with_osv_best = [
        (x + (medians[i],))
        for i, chunk in enumerate(chunks)
        for x in chunk
    ]
    return all_overshoot_with_osv_best


def compute_all_overshoot(
    prices: List[float],
    upturn: List[Tuple[int, float]],
    downturn: List[Tuple[int, float]],
    p_ext: List[Tuple[float, int, str]],
    threshold: float,
) -> List[Tuple[int, float, float, str]]:
    """
    Compute all overshoot values based on given parameters.

    Args:
    - prices: List of price values.
    - upturn: List of upturn events.
    - downturn: List of downturn events.
    - p_ext: List of price extension events.
    - threshold: Threshold value for calculation.

    Returns:
    - List of tuples containing overshoot data.
    """
    all_overshoot = []
    up_cursor, down_cursor = 0, 0
    for i, (price, index, event) in enumerate(p_ext):
        p_dcc = price * (
            1 + threshold
        )  # From Pdcc = Pext . (1 + THETA)
        stop = (
            upturn[up_cursor][0]
            if event == "UR"
            else downturn[down_cursor][0]
        )
        if event == "UR":
            up_cursor += 1
        else:
            down_cursor += 1
        for j in range(index, stop):
            osv_cur = (prices[j] / p_dcc) / (threshold * p_dcc)
            all_overshoot.append((j, prices[j], osv_cur, event))
    return all_overshoot


def split_into_chunks(
    data: List[Tuple[int, float, float, str]], chunk_size: int
) -> List[List[Tuple[int, float, float, str]]]:
    """
    Split data into chunks of specified size.

    Args:
    - data: List of data to be split.
    - chunk_size: Size of each chunk.

    Returns:
    - List of chunks.
    """
    return [
        data[i : i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]
