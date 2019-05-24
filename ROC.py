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
from SentimentApp.tracker.models import Review


sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp')
sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp\SentimentApp')
sys.path.append(r'C:\Users\j.turnbull\PycharmProjects\SentimentApp') # add path to project root dir
os.environ["DJANGO_SETTINGS_MODULE"] = "SentimentApp.settings"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SentimentApp.settings')

# for more sophisticated setups, if you need to change connection settings (e.g. when using django-environ):
#os.environ["DATABASE_URL"] = "postgres://myuser:mypassword@localhost:54324/mydb"
settings.configure()
# Connect to Django ORM
django.setup()

# process data
# Review.objects.create(reviewText='MyAgency', predictSentiment='POSITIVE', actualSentiment='POSITIVE')
from tracker.models import Review
from tracker.models import PosScores
from tracker.models import NegScores
from tracker.models import WeightedAvg


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
    auc = roc_auc_score(y_pred, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_pred, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.show()


getROC()