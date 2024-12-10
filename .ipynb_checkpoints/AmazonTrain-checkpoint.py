import pickle

# from sklearn.datasets import load_files
import numpy as np
import urllib
import pandas as pd
import re
# import mglearn
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
import time
import dataprocess as proc
import machinealgs as algs
import os
import sys
import spacy
# import nltk

##sklearn funcs
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing, model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation


start = time.time()


path = os.getcwd()

print(path)
directory = path + '/DATA/'
print(directory)
singleJSON = directory+'output.json.gz'


## loading datasets to pandas dataframe
singleDF = proc.getDF(singleJSON)
df = proc.files2DF(singleDF, directory)
# df = proc.getDF(gzpathSmall)
df = df.reset_index(drop=True)

print(df.head())
print(proc.sentiCounts(df, 'sentiment'))
print(proc.sentiCounts(df, 'rating'))
df.info(memory_usage='deep')

# sys.exit("Page Break")
## DATA SPLIT
# reviews_with_id = df.reset_index() # adds an `index` column
# reviewText = df['reviewText']
# train_set1, test_set2 = proc.split_train_test_by_id(reviews_with_id, 0.2, "index") # custom script that, splits dataset 80:20
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #splits dataset 80:20, always produces the isolated test set
print(proc.sentiCounts(train_set, 'rating'))

# balancing dataframe
# sample_size = 53000
sample_size = 5000
# sample_size = 200
train_set = proc.balanceData(singleDF, train_set, sample_size)

train_set = proc.addSenimentColumn(train_set, 'd1')
test_set = proc.addSenimentColumn(test_set, 'd1')

test_set.to_csv(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp\Datasets\TestData\TestData.csv', encoding='utf-8', index=False)
print("Test data uploaded")

# sys.exit("stopped")
text_test, y_test = test_set['reviewText'], test_set['sentiment'] # test_set['overall'] provides labels
text_train = train_set['reviewText']    # easier naming convention

# train_set.to_csv(train_set, encoding='utf-8', index=False)

## using the bag of words model
vectorizer = CountVectorizer(max_features=10000)
# vectorizer = HashingVectorizer()
vect = vectorizer.fit(text_train)
X_train = vect.transform(text_train)
y_train = train_set['sentiment']
print("X_train:\n{}".format(repr(X_train))) # outputs the matrix representation
X_test = vect.transform(text_test)


# using the get_feature_name method to access the vocabulary, identifying features
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 2010 to 2030:\n{}".format(feature_names[2010:2030]))
print("Every 500th feature:\n{}".format(feature_names[::500]))


# algs.batchAlgs(X_train, y_train, X_test, y_test)
# sys.exit('break')
# hyperparameter tuning
algs.top4(X_train, y_train, X_test, y_test)
sys.exit("sterp")
# algs.top2(X_train, y_train, X_test, y_test)
# algs.top2Tuning(X_train, y_train, X_test, y_test)


# vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3), stop_words='english')
# vect = vectorizer.fit(text_train)
# X_train = vect.transform(text_train)
# vect = vectorizer.fit(text_train)
# X_train = vect.transform(text_train)

# stop words
# Specifying stop_words="english" uses the built-in list - can pass own list of stop words
# algs.stopWordRemoval(text_train, y_train)


# # using pipeline to ensure the results of grid search are valid
## pipelines, tdidf layout and histogram of results
# algs.tdidfResults(text_train, y_train)

# ## trigrams
# algs.trigramModel(text_train, y_train)

# hyperparamter tuning n-grams
## producing heat map of 1-3 grams and coefficent models ####
# algs.ngramModel(text_train, y_train)

## custom vectorizer using spacy
# algs.customerVectorizer(text_train, y_train)

### LDA implementation with bar charts
# algs.ImplementLDA(text_train)

# pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# # running the grid search takes a long time because of the
# # large grid and the inclusion of trigrams
# param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
#               "logisticregression__solver": ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
#               "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
#               "tfidfvectorizer__stop_words": ['english']
#               }
#
# grid = GridSearchCV(pipe, param_grid, cv=5)
# grid.fit(text_train, y_train)
# print("Best cross-validation score: {:.2f}".format(grid.best_score_))
# print("Best parameters:\n{}".format(grid.best_params_))

# pipe = make_pipeline(TfidfVectorizer(min_df=5), SVC())
#
# param_grid = [
#     {'SVC__C': [1, 10, 100, 1000], 'SVC__kernel': ['linear'],
#      "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]},
#     {'SVC__C': [1, 10, 100, 1000], 'SVC__gamma': [0.001, 0.0001], 'SVC__kernel': ['rbf']},
# ]
#
# grid = GridSearchCV(pipe, param_grid, cv=5)
# grid.fit(X_train, y_train)
# print("Best cross-validation SVM score: {:.2f}".format(grid.best_score_))
# print("Best parameters: ", grid.best_params_)

vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3), stop_words='english')
tfidf = vectorizer.fit(text_train)
X_train = tfidf.transform(text_train)
X_test = tfidf.transform(text_test)

# lg = LogisticRegression(C=1, solver='liblinear')
# lg.fit(X_train, y_train)
# print("LogReg Accuracy on training set: {:.3f}".format(lg.score(X_train, y_train)))
# print("LogReg Accuracy on test set: {:.3f}".format(lg.score(X_test, y_test)))


sgd = SGDClassifier(loss='log', penalty='l2', n_iter=1000)
sgd.fit(X_train, y_train)
print("SGD/LogReg Accuracy on training set: {:.3f}".format(sgd.score(X_train, y_train)))
print("SGD/LogReg Accuracy on test set: {:.3f}".format(sgd.score(X_test, y_test)))


# # save the model to disk
path = os.getcwd()
transformerPath = path + '/savedModels/transformer/'
modelPath = path + '/savedModels/model/'

tfidf_filename = transformerPath+'finalised_tfidftransformer.sav'
pickle.dump(tfidf, open(tfidf_filename, 'wb'))
#
model_filename = modelPath+'finalised_model.sav'
pickle.dump(sgd, open(model_filename, 'wb'))


end = time. time()
print(end - start)


