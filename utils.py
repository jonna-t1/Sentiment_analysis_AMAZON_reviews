import pandas as pd
import numpy as np

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
    count = df.groupby(sentiCut,observed=True).overall.count().reset_index(name='count')

  if option == 'rating' or option == 'Rating':
    count = df.groupby(ratingCut,observed=True).overall.count().reset_index(name='count')

  return count
def produceSample(train_set, labelNum, sample_size):

  np.random.seed(123)  ## this was used to produce the same every time

  rating_indices = train_set[train_set.overall == labelNum].index
  random_indices = np.random.choice(rating_indices, sample_size, replace=False)
  rating1_sample = train_set.loc[random_indices]
  return rating1_sample
def balanceData(train_set, sample_size):

    rating5_sample = produceSample(train_set, 5, sample_size)
    rating4_sample = produceSample(train_set, 4, sample_size)
    rating3_sample = produceSample(train_set, 3, sample_size)
    rating2_sample = produceSample(train_set, 2, sample_size)
    rating1_sample = produceSample(train_set, 1, sample_size)

    train_set = pd.DataFrame()

    train_set = pd.concat([rating5_sample, rating4_sample,rating3_sample,rating2_sample,rating1_sample], ignore_index=True)

    print("Dataset balanced...")
    print(sentiCounts(train_set, 'rating'))
    return train_set