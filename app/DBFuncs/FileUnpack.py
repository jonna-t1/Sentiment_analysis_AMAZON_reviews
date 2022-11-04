import gzip
import os
import pandas as pd
import json

def parse(path):
  # print(path)
  g = gzip.open(path, 'rb')
  for l in g:
      # print(type(l))
      yield json.loads(l)

def getDF(path,*args):
  i = 0
  df = {}
  for d in parse(path):
      df[i] = d
      i += 1

  if args == 'json':
    out = open(path, "r")
    for d in out:
      df[i] = d
      i += 1

  # print(sys.getsizeof(df))
  return pd.DataFrame.from_dict(df, orient='index')


def files2DF(df, directory):
  count = 0
  file_list = []
  for filename in os.listdir(directory):
      if filename.endswith(".gz"):
        if filename == "output.json.gz":
          continue
        else:
          # print(filename)
          filepath = directory+filename
          file_list.append(filepath)
  return file_list

def df_l(arr):
  count = 0
  df_arr = []
  for f_path in arr:    
    count+=1
    dataframe = getDF(f_path, 'gz')
    df_arr.append(dataframe)
    print(f"File: {f_path} added to dataframe array.")
  return df_arr