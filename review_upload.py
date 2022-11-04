import os
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from app.DBFuncs.dbUtils import postgres_remove_dups, str2float, addSenimentColumn
import app.DBFuncs.dbConfig as db

from DBFuncs import FileUnpack as Unpack

def _open_file():
        paths = fd.askopenfilenames(filetypes=[('.gz files', '*.gz'),
                                                       ('all files', '.*')],
                                            initialdir="./DATA/",
                                            title="Select files", multiple=True)
        if not paths or paths == '':
            return
        return paths

try:
    root = tk.Tk()
    root.withdraw()
    files_to_upload = _open_file()
    file_paths = np.asarray(files_to_upload)
except:
    print("No files selected")

files_size = 0
for i in file_paths:
    files_size += os.stat(i).st_size
    if files_size > int(5e8):
        print("Combined File size exceeds allowed quota. Try a smaller file sizes.")
        break

print("Combining df's...")
df_arr = Unpack.df_l(file_paths)
df = pd.concat(df_arr, axis=0, sort=False)

print(df.shape)
df = df[["overall","vote","verified","reviewText","summary"]]
df = df.rename(columns={'reviewText': 'reviewtext'})
df = str2float(df)    #convert string to float in ratings column
df = addSenimentColumn(df, 'd1')  # add sentiment column

print(df.columns)

database=db.config["database"]
user=db.config["user"]
password=db.config["password"]
host=db.config["host"]
port= db.config["port"]
tbl = db.config["tbl"]

try:
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
    df.to_sql(f'{tbl}', engine, if_exists='append', index=False)
    print("Data from files added to database.")
except Exception as e:
    print(e)

try:
    postgres_remove_dups()
    print("Removing duplicates..")
except Exception as e:
    print("\nDrop rows failed\n")
    print(e)
