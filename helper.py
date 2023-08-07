# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:22:21 2023

@author: sheeh
"""

import numpy as np

def get_df_ohlc(df, frequency):
    df[f'group_id_{frequency}'] = df.index // frequency
    grouped = df.groupby(f'group_id_{frequency}')
    price_ohlc = grouped['mid_price'].ohlc()
    
    return price_ohlc


def split_data(dataset, split_percent):
    split_point = int(len(dataset) * split_percent)
    return dataset[0:split_point, :], dataset[split_point:len(dataset), :]


def transform_data_for_NN(dataset, seq_size):
    x = []
    y = []
    
    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)