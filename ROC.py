import os
import re
import sys

import django
from django_pandas.io import read_frame
from matplotlib import pyplot

import retrainModel as retrain
import pandas as pd
import pickle
import dataprocess as proc
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import ModelUtils as utils


def getROC():
    path = os.getcwd()
    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'

    # load the model from disk
    tfidf_filename = utils.getLastModel(transformerPath)
    model_filename = utils.getLastModel(modelPath)
    loaded_tfidf = pickle.load(open(tfidf_filename, 'rb'))
    loaded_model = pickle.load(open(model_filename, 'rb'))


    ## get from db
    qs = Review.objects.all()
    reviews = read_frame(qs)

    ## loading dummy data
    data = reviews['reviewText']

    ## model prediction on new data
    X_new = loaded_tfidf.transform(data)
    y_pred = loaded_model.predict(X_new)

    target_names = ['positive', 'negative']
    y_true = reviews['sentiment']


    probs = loaded_model.predict_proba(X_new)

    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate AUC
    auc = roc_auc_score(y_true, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_pred, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.show()



def test():
    f = open("guru99.txt", "w+")


try:
    test()
except NameError as err:
    print("shat")