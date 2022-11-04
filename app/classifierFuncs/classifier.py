from sqlalchemy import create_engine
import sys
sys.path.append("../../") # Adds higher directory to python modules path.
import DBFuncs.dbConfig as db
import pandas as pd
from . import procs
from sklearn.model_selection import train_test_split, KFold

user = db.config['user']
host = db.config['host']
port = db.config['port']
database = db.config['database']
tbl = db.config['tbl']

def fetch_balance(n, sample_size):
    """
    Retrieve the top n rows, splits the data into train and test and balances the dataset.
    """
    records = 1000
    ce = f'postgresql://{user}@{host}:{port}/{database}'
    df = pd.DataFrame()
    try:
        engine = create_engine(ce)
        df = pd.read_sql_query(f'select * from "{tbl}" order by id desc limit {n}',con=engine)
    except Exception as e:
        print(e)

    df = procs.str2float(df)

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #splits dataset 80:20, always produces the isolated test set
    counts = procs.sentiCounts(train_set, 'rating')
    min_count = counts["count"].min()

    ## check if sample size is less than the minimum count of the rows returned
    if sample_size > counts["count"].min():
        print(counts)
        print("Min count value: "+str(min_count))
        print("\n### FAILURE: ###\nProvide a smaller sample size, OR increase number of rows returned.\n")
        exit()

    train_set = procs.balanceData(train_set, sample_size)
    return train_set, test_set
    # print(procs.sentiCounts(train_set, 'rating'))

def get_info(vect):
    """
    Prints CountVectorizer information; \n
    1. no. of features, 2. first 20 features, 3. Features 2010 to 2030, 4. Every 500th feature.
    """
    # using the get_feature_name method to access the vocabulary, identifying features
    feature_names = vect.get_feature_names_out()
    print("Number of features: {}".format(len(feature_names)))
    print("First 20 features:\n{}".format(feature_names[:20]))
    print("Features 2010 to 2030:\n{}".format(feature_names[2010:2030]))
    print("Every 500th feature:\n{}".format(feature_names[::500]))