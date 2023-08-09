# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from collections import defaultdict
from pandas import DatetimeIndex
import numpy as np
import pandas as pd
from helper.dc import calculate_dc


# Global Variables

THETA_THRESHOLDS = (
    np.array([0.098, 0.22, 0.48, 0.72, 0.98, 1.22, 1.55, 1.70, 2, 2.55]) / 100
)
THETA_WEIGHTS = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.05, 0.1, 0.08, 0.07, 0.1])


# Read the data from CSV
df = pd.read_csv("stock_data.csv")

timestamps = df["Date"]

ALL_STOCK = df["ALL"]

upturn, downturn, p_ext = calculate_dc(ALL_STOCK, 0.098 / 100)


def calculate_dc_indicators(prices, upturn, downturn, p_ext, threshold, chunk_size=4):
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
                all_overshoot.append((j, prices[j], osv_cur, p_ext[i][2]))
            continue

        if (
            p_ext[i][2] == "UR"
            and downturn[down_cursor - 1][0] != p_ext[i][1]
            or p_ext[i][2] == "DR"
            and upturn[up_cursor - 1][0] != p_ext[i][1]
        ):
            for j in range(p_ext[i][1], stop):
                osv_cur = (prices[j] / p_dcc) / (threshold * p_dcc)
                all_overshoot.append((j, prices[j], osv_cur, p_ext[i][2]))
    
    # Split the array into chunks
    chunks = [all_overshoot[i:i + chunk_size] for i in range(0, len(all_overshoot), chunk_size)]
    
    # Calculate median for each chunk
    medians = [np.median([x[2] for x in chunk]) for chunk in chunks]
    
    # Create new array with tuples
    all_overshoot_with_osv_best = []
    
    for i, chunk in enumerate(chunks):
        for x in chunk:
            all_overshoot_with_osv_best.append(x + (medians[i],))
    
    return all_overshoot_with_osv_best


all_overshoot_with_osv_best = calculate_dc_indicators(ALL_STOCK, upturn, downturn, p_ext, 0.098 / 100) 


def run_strategy_1(thresholds, weights, prices):
    threshold_dc_summaries = []
    for threshold in thresholds:
        upturn, downturn, p_ext = calculate_dc(prices, threshold)
        upturn = [x + ('UR',) for x in upturn]
        downturn = [x + ('DR',) for x in downturn]
        
        dc = []
        if len(upturn) > len(downturn):
            for a, b in zip(upturn, downturn):
                dc.extend([a, b])
        else:
            for a, b in zip(downturn, upturn):
                dc.extend([a, b])

        threshold_dc_summaries.append({
            "dc": threshold_dc_summaries,
            "p_ext": p_ext
        })
    
    thresholds_last_dc_index = [-1] * len(thresholds)
    thresholds_last_dc_time_interval = [-1] * len(thresholds)
    thresholds_decision = ['h'] * len(thresholds)
    current_position = 'h'
    
    # def update_threshold_choices(i):
        
    
    for i in range(len(prices)):
        for j in range(len(thresholds)):
            next_dc_index = -1
            current_threshold_summary = threshold_dc_summaries[j]
            
            for n in current_threshold_summary['dc']:
                # This price index matches a dc confirmation point for this threshold
                if n[1] == i:
                    thresholds_last_dc_index[j] = n[1]
                    last_ext = -1
                    for e in current_threshold_summary['p_ext']:
                        
        
        
    return threshold_dc_summaries
    
    
    
z= run_strategy_1(THETA_THRESHOLDS, THETA_WEIGHTS, ALL_STOCK)
    
    
    
    