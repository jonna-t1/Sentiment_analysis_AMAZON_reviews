import sys
import os
import numpy as np
import hashlib
import pandas as pd
import gzip
import time

def timer(func):
    start_time = time.time()
    func()
    print("--- %s seconds ---" % (time.time() - start_time))

def parse(path):
    # print(path)
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path, **args):
    print(path)
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

    print(sys.getsizeof(df))
    return pd.DataFrame.from_dict(df, orient='index')

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def sentiCounts(df, option):
    sentiCut = pd.cut(
        df.overall,
        [0, 2.5, np.inf],
        labels=['negative', 'positive']
    )
    ratingCut = pd.cut(
        df.overall,
        [0, 1.0, 2.0, 3.0, 4.0, np.inf],
        labels=['1', '2', '3', '4', '5']
    )

    if option == 'sentiment' or option == 'Sentiment':
        count = df.groupby(sentiCut).overall.count().reset_index(name='count')

    if option == 'rating' or option == 'Rating':
        count = df.groupby(ratingCut).overall.count().reset_index(name='count')

    return count

class style():
    BLACK = lambda x: '\033[30m' + str(x)
    RED = lambda x: '\033[31m' + str(x)
    GREEN = lambda x: '\033[32m' + str(x)
    YELLOW = lambda x: '\033[33m' + str(x)
    BLUE = lambda x: '\033[34m' + str(x)
    MAGENTA = lambda x: '\033[35m' + str(x)
    CYAN = lambda x: '\033[36m' + str(x)
    WHITE = lambda x: '\033[37m' + str(x)
    UNDERLINE = lambda x: '\033[4m' + str(x)
    RESET = lambda x: '\033[0m' + str(x)

def produceSample(train_set, labelNum, sample_size):

    np.random.seed(123)     ## this was used to produce the same every time

    rating_indices = train_set[train_set.overall == labelNum].index
    random_indices = np.random.choice(rating_indices, sample_size, replace=False)
    rating1_sample = train_set.loc[random_indices]
    return rating1_sample


def balanceData(singleDF, train_set, sample_size):

    rating5_sample = produceSample(train_set, 5, sample_size)
    rating4_sample = produceSample(train_set, 4, sample_size)
    rating3_sample = produceSample(train_set, 3, sample_size)
    rating2_sample = produceSample(train_set, 2, sample_size)
    rating1_sample = produceSample(train_set, 1, sample_size)

    train_set = singleDF

    train_set = train_set.append(rating5_sample)
    train_set = train_set.append(rating4_sample)
    train_set = train_set.append(rating3_sample)
    train_set = train_set.append(rating2_sample)
    train_set = train_set.append(rating1_sample)

    print(sentiCounts(train_set, 'rating'))
    return train_set

def files2DF(df, directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".gz"):
            filepath = directory+"\\"+filename

            dataframe = getDF(filepath, 'gz')
            df = df.append(dataframe)
            count = count + 1
            print(count)

            continue
        else:
            continue
    return df

def addSenimentColumn(train_set, option):

    if option == 'd1':
        positive_rating_indices = train_set[train_set.overall > 2].index
        negative_rating_indices = train_set[train_set.overall <= 2].index

        train_set['sentiment'] = 'positive'
        train_set['sentiment'].loc[negative_rating_indices] = 'negative'

        # print(train_set.loc[positive_rating_indices].head())
        # print(train_set.loc[negative_rating_indices].head())

    if option == 'd2':
        positive_rating_indices = train_set[train_set.reviewsrating > 2].index
        negative_rating_indices = train_set[train_set.reviewsrating <= 2].index

        train_set['sentiment'] = 'positive'
        train_set['sentiment'].loc[negative_rating_indices] = 'negative'

        # print(train_set.loc[positive_rating_indices].head())
        # print(train_set.loc[negative_rating_indices].head())

    return train_set