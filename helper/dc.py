def calculate_dc(prices, threshold):
    """
    Calculate Donchian Channels (DC) based on given price data and threshold.
    
    Parameters:
    - prices (list): A list of price data.
    - threshold (float): A threshold value to determine upturns and downturns.
    
    Returns:
    - upturn, downturn, extreme price points
    """
    
    last_low_index = 0
    last_high_index = 0
    last_low_price = prices[last_low_index]
    last_high_price = prices[last_high_index]
    current_index = 1
    
    upturn_dc = []
    downturn_dc = []
    p_ext = []
    
    event = None
    
    # Determine the initial event
    while current_index < len(prices):
        if prices[current_index] <= last_high_price * (1 - threshold):
            downturn_dc.append((current_index, prices[current_index]))
            p_ext.append((last_high_price, last_high_index, 'DR'))
            event = 'DR'
            break
        elif prices[current_index] >= last_low_price * (1 + threshold):
            upturn_dc.append((current_index, prices[current_index]))
            p_ext.append((last_low_price, last_low_index, 'UR'))
            event = 'UR'
            break
        elif prices[current_index] > last_high_price:
            last_high_index = current_index
            last_high_price = prices[current_index]
        elif prices[current_index] < last_low_price:
            last_low_index = current_index
            last_low_price = prices[current_index]
        current_index += 1
    
    # Determine subsequent events
    while current_index < len(prices):
        if event == 'DR':
            if prices[current_index] < last_low_price:
               last_low_index = current_index
               last_low_price = prices[current_index]
            elif prices[current_index] >= last_low_price * (1 + threshold):
                upturn_dc.append((current_index, prices[current_index]))
                p_ext.append((last_low_price, last_low_index, 'UR'))
                last_high_index = current_index
                last_high_price = prices[current_index]
                event = 'UR'
        elif event == 'UR':
            if prices[current_index] > last_high_price:
                last_high_index = current_index
                last_high_price = prices[current_index]
            elif prices[current_index] <= last_high_price * (1 - threshold):
                downturn_dc.append((current_index, prices[current_index]))
                p_ext.append((last_high_price, last_high_index, 'DR'))
                event = 'DR'
                last_low_index = current_index
                last_low_price = prices[current_index]
        current_index += 1
    
    return upturn_dc, downturn_dc, p_ext