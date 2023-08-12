import pandas as pd
from db import Database

db = Database()
# get the data into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM gbpusd;", db.engine)
df = df.drop('vol', axis=1)
df.to_csv('data/gbpusd.csv', sep='\t', encoding='utf-8')
print("Created gbpusd csv")

df = pd.read_sql_query("SELECT * FROM eurusd;", db.engine)
df = df.drop('vol', axis=1)
df.to_csv('data/eurusd.csv', sep='\t', encoding='utf-8')
print("Created eurusd csv")

df = pd.read_sql_query("SELECT * FROM usdchf;", db.engine)
df = df.drop('vol', axis=1)
df.to_csv('data/usdchf.csv', sep='\t', encoding='utf-8')
print("Created usdchf csv")

df = pd.read_sql_query("SELECT * FROM usdcad;", db.engine)
df = df.drop('vol', axis=1)
df.to_csv('data/usdcad.csv', sep='\t', encoding='utf-8')
print("Created usdcad csv")

#%%
