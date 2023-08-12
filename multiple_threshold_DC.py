import numpy as np
import pandas as pd
from helper.dc import calculate_dc


def calculate_dc_indicators(
    prices, upturn, downturn, p_ext, threshold, chunk_size=4
):
    all_overshoot = []
    up_cursor = 0
    down_cursor = 0
    for i in range(len(p_ext)):
        p_dcc = p_ext[i][0] * (1 + threshold)

        if p_ext[i][2] == "UR":
            stop = upturn[up_cursor][0]
            up_cursor += 1
        else:
            stop = downturn[down_cursor][0]
            down_cursor += 1

        if i == 0:
            for j in range(p_ext[i][1], stop):
                osv_cur = (prices[j] / p_dcc) / (threshold * p_dcc)
                all_overshoot.append(
                    (j, prices[j], osv_cur, p_ext[i][2])
                )
            continue

        if (
            p_ext[i][2] == "UR"
            and downturn[down_cursor - 1][0] != p_ext[i][1]
            or p_ext[i][2] == "DR"
            and upturn[up_cursor - 1][0] != p_ext[i][1]
        ):
            for j in range(p_ext[i][1], stop):
                osv_cur = (prices[j] / p_dcc) / (threshold * p_dcc)
                all_overshoot.append(
                    (j, prices[j], osv_cur, p_ext[i][2])
                )

    # Split the array into chunks
    chunks = [
        all_overshoot[i : i + chunk_size]
        for i in range(0, len(all_overshoot), chunk_size)
    ]

    # Calculate median for each chunk
    medians = [np.median([x[2] for x in chunk]) for chunk in chunks]

    # Create new array with tuples
    all_overshoot_with_osv_best = []

    for i, chunk in enumerate(chunks):
        for x in chunk:
            all_overshoot_with_osv_best.append(x + (medians[i],))

    return all_overshoot_with_osv_best


def calculate_decision(thresholds, thresholds_last_decision, weights):
    decisions = [("s", 0), ("h", 0), ("b", 0)]
    for j, threshold in enumerate(thresholds):
        if thresholds_last_decision[j] == "b":
            decisions[2] = ("b", decisions[2][1] + weights[j])
        elif thresholds_last_decision[j] == "s":
            decisions[0] = ("s", decisions[0][1] + weights[j])
        else:
            decisions[1] = ("h", decisions[1][1] + weights[j])

    return max(decisions, key=lambda x: x[1])[0]


def run_strategy_1(thresholds, weights, prices):
    threshold_dc_summaries = []
    for threshold in thresholds:
        upturn, downturn, p_ext = calculate_dc(prices, threshold)
        upturn = [x + ("UR",) for x in upturn]
        downturn = [x + ("DR",) for x in downturn]

        dc_indexes = []
        dc_prices = []
        dc_event = []
        if len(upturn) > len(downturn):
            for a, b in zip(upturn, downturn):
                dc_indexes.extend([a[0], b[0]])

                dc_prices.extend([a[1], b[1]])

                dc_event.extend([a[2], b[2]])
        else:
            for a, b in zip(downturn, upturn):
                dc_indexes.extend([a[0], b[0]])

                dc_prices.extend([a[1], b[1]])

                dc_event.extend([a[2], b[2]])

        threshold_dc_summaries.append(
            {
                "dc": pd.DataFrame(
                    data={
                        "price": np.array(dc_prices),
                        "event": dc_event,
                    },
                    index=dc_indexes,
                ),
                "p_ext": pd.DataFrame(
                    data={
                        "price": [x[0] for x in p_ext],
                        "event": [x[2] for x in p_ext],
                    },
                    index=[x[1] for x in p_ext],
                ),
            }
        )

    thresholds_last_dc = [None] * len(thresholds)
    thresholds_last_dc_time_interval = [-1] * len(thresholds)

    thresholds_last_decision = [None] * len(thresholds)

    # Positions can be 'b', 'h', 's'
    last_position = None

    last_buy_price = 0

    return_prices = pd.DataFrame(
        {
            "prices": prices,
        }
    )
    return_prices["returns"] = 0

    for i in range(len(prices)):
        for j, threshold in enumerate(thresholds):
            dc = threshold_dc_summaries[j]["dc"]
            p_ext = threshold_dc_summaries[j]["p_ext"]

            if i in dc.index:
                # DC event found for this price index
                last_price_ext = p_ext[
                    p_ext.index < i
                ].last_valid_index()
                time_interval_to_dc = i - last_price_ext

                thresholds_last_dc[j] = dc.loc[i]
                thresholds_last_dc_time_interval[
                    j
                ] = time_interval_to_dc

            else:
                if not isinstance(thresholds_last_dc[j], pd.Series):
                    continue

                if thresholds_last_dc_time_interval[j] == (
                    i - thresholds_last_dc[j].name
                ):
                    if (
                        thresholds_last_dc[j]["event"] == "DR"
                        and last_position != "b"
                    ):
                        thresholds_last_decision[j] = "b"
                    elif (
                        thresholds_last_dc[j]["event"] == "UR"
                        and last_position != "s"
                    ):
                        thresholds_last_decision[j] = "s"
                else:
                    thresholds_last_decision[j] = "h"
        new_position = calculate_decision(
            thresholds,
            thresholds_last_decision,
            weights,
            last_position,
        )

        if new_position != "h":
            dr_count = 0
            for td in thresholds_last_dc:
                if isinstance(td, pd.Series) and td["event"] == "DR":
                    dr_count += 1
            ur_count = 12 - dr_count
            if (
                new_position == "b"
                and last_position != "b"
                and dr_count > ur_count
            ):
                last_buy_price = prices[i] * 1.0025
                last_position = new_position
            elif (
                new_position == "s"
                and last_position != "s"
                and dr_count < ur_count
            ):
                return_prices.loc[i, "returns"] = (
                    last_buy_price - prices[i]
                )
                # return_prices["returns"][i] = last_buy_price - prices[i]
                last_position = new_position

    return return_prices


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


# THETA_WEIGHTS = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.05, 0.1, 0.08, 0.07, 0.1])

# # Read the data from CSV
# df = pd.read_csv("stock_data.csv")


# THETA_THRESHOLDS = (
#     np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
# )

# XEL = df['XEL']

# run = run_strategy_1(THETA_THRESHOLDS, THETA_WEIGHTS, XEL)
