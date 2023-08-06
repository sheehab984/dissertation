import psycopg2
import pandas as pd
from db import Database




if __name__ == "__main__":
    db = Database()
    # get the data into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM gbpusd;", db.engine)

    # after getting the data, we can calculate the OHLC for each 100 trades
    df['group_id'] = df.index // 1000
    bid_ohlc = df.groupby('group_id')['bid_price'].ohlc()
    ask_ohlc = df.groupby('group_id')['ask_price'].ohlc()

    