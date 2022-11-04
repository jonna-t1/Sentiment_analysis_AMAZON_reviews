import os

import pandas as pd
import psycopg2

from app.DBFuncs import FileUnpack as Unpack

# define path to data here
data_dir = os.getcwd() + "/DATA/"
singleJSON = data_dir+'output.json.gz'

## loading datasets to pandas dataframe
singleDF = Unpack.getDF(singleJSON)

try:
    interested_cols = singleDF[['overall','vote','verified','reviewText','summary']].columns.to_list()
    # summary_col = interested_cols[-1]
    # del interested_cols[-1]
    print(interested_cols)
except:
    print("\nDataset does not contain those columns.\n")

### Defines db table creation
q_str1 = "CREATE TABLE Reviews (id SERIAL PRIMARY KEY, "
cols_str = ""
for i in interested_cols:
    cols_str += f"{i} VARCHAR(255) NOT NULL, "
# query_str = q_str1+cols_str+f"{summary_col} VARCHAR(255) NOT NULL)"
'overall','vote','verified','reviewText','summary'
query_str = "CREATE TABLE Reviews (id SERIAL PRIMARY KEY,\
                overall VARCHAR(255),\
                vote VARCHAR(255),\
                verified VARCHAR(255),\
                reviewText TEXT,\
                sentiment VARCHAR(255),\
                summary VARCHAR(255))"


def postgres_create_tbl():
    try:
        conn = psycopg2.connect(
          database="Review", user='postgres', password='test', host='127.0.0.1', port= '5432'
        )
        cur = conn.cursor()
        cur.execute(query_str)
        cur.close()
        conn.commit()
        return "Database created without issue!"
    except Exception as e:
        return f"{e}"

print(postgres_create_tbl())
