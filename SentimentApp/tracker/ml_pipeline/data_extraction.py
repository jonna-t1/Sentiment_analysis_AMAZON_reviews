import os
import gzip
import json
import pandas as pd

def get_most_recent_file(data_directory):
    # Ensure the directory exists
    if not data_directory.exists() or not data_directory.is_dir():
        raise FileNotFoundError(f"Directory {data_directory} does not exist.")

    # Get all files in the 'DATA' directory
    files = list(data_directory.glob("*"))

    # Check if the directory is empty
    if not files:
        raise FileNotFoundError(f"No files found in {data_directory}")

    # Get the most recent file by modification time
    most_recent_file = max(files, key=lambda f: f.stat().st_mtime)
    print(most_recent_file)
    return most_recent_file
    # Get the most recent file based on modification time
    most_recent_file = max(files, key=os.path.getmtime)
    return most_recent_file

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')