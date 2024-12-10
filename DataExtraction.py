import pandas as pd
import gzip
import json
import os
import numpy as np

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


def process_directory_to_dataframe(directory):
  """
  Iterate through all `.gz` files in a directory and merge their contents into a single DataFrame.

  Args:
      directory (str): Path to the directory containing `.gz` files.

  Returns:
      pd.DataFrame: Combined DataFrame with all files' data.
  """
  combined_df = pd.DataFrame()  # Initialize an empty DataFrame

  for file_name in os.listdir(directory):
    if file_name.endswith(".gz"):  # Process only `.gz` files
      file_path = os.path.join(directory, file_name)
      print(f"Processing file: {file_path}")
      try:
        file_df = getDF(file_path)  # Convert file content to a DataFrame
        combined_df = pd.concat([combined_df, file_df], ignore_index=True)  # Append to the combined DataFrame
      except Exception as e:
        print(f"Error processing {file_path}: {e}")

  return combined_df
