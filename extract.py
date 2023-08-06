import os
import glob
import zipfile
import pandas as pd
from db import Database
import subprocess

db = Database()

current_script_path = os.path.abspath(__file__)
current_path = os.getcwd()

def process_csv_data(csv_content, file_name, directory):
    # Slice the DataFrame to get the first three columns
    table_name = file_name.strip('.csv').split('_')[2].lower()

    absolute_path = os.path.abspath(file_name)

    file_content = f'''COPY {table_name}(timestamp, bid, ask, vol) 
FROM '{absolute_path}' 
DELIMITER ',' 
CSV;'''

    db.createTable(table_name)

    with open(os.path.join(current_path, 'script.sql'), 'w') as file:
        file.write(file_content)
    
    subprocess.run(["psql", "-h", "localhost", "-U", "postgres", "-d", "postgres", "-f", os.path.join(current_path, 'script.sql')]) 

    print(f"Finished processing file: {file_content}")

def main(directory):
    os.chdir(directory)

    # Unzip all the zip files in the directory.
    for file in glob.glob("*.zip"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(directory)
    
    # Process all the csv files in the directory.
    for csv_file in glob.glob(os.path.join(directory, "*.csv")):
        data = pd.read_csv(csv_file)
        process_csv_data(data, csv_file, directory)

            
if __name__ == "__main__":
    # directory = './Dataset/USDCHF'  # update this with your directory path
    # main(directory)
    pass