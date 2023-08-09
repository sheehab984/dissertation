# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_price_with_highlight(price_data, zipped_data):
    """
    Plot the full price data and highlight specific segments using seaborn.

    Parameters:
    - price_data (list): A list of price data.
    - zipped_data (list of tuples): A list of tuples where each tuple contains the start and end index to be highlighted.
    """
    
    # Create a DataFrame from the price data
    df = pd.DataFrame(price_data, columns=['Price'])
    
    # Create a new column 'Highlight' to determine which points should be highlighted
    df['Highlight'] = 0
    for start, end in zipped_data:
        df['Highlight'].iloc[start:end+1] = 1
    
    # Plot the data with improved DPI and adjusted line visibility
    plt.figure(figsize=(12, 6), dpi=400)
    
    # Plot the main price data with reduced visibility
    sns.lineplot(data=df, y='Price', x=df.index, color='blue', linewidth=1, label='Price Data')
    
    # Plot the highlighted segments with increased prominence
    highlighted = df[df['Highlight'] == 1]
    sns.lineplot(data=highlighted, y='Price', x=highlighted.index, color='red', alpha=0.5, linewidth=1, label='Highlighted Segment')
    
    plt.title('Price Data with Highlighted Segments')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    return df

# zipped_data = [(a[0], b[0]) for a, b in zip(upturn, downturn)] if upturn[0] < downturn[0] else [(a[0], b[0]) for a, b in zip(downturn, upturn)]
# x = plot_price_with_highlight(ALL_STOCK.to_numpy(), zipped_data)
