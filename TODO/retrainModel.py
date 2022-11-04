
import os
from tkinter import Tk, filedialog
import pandas as pd
import pickle
import sys
import numpy as np
import dataprocess as proc
from colorama import Fore
import ModelUtils

import django
from django_pandas.io import read_frame
from matplotlib import pyplot

import retrainModel as retrain
import pandas as pd
import pickle
import dataprocess as proc
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import ModelUtils as utils


sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp')
sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp\SentimentApp')
sys.path.append(r'C:\Users\j.turnbull\PycharmProjects\SentimentApp') # add path to project root dir
os.environ["DJANGO_SETTINGS_MODULE"] = "SentimentApp.settings"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SentimentApp.settings')

# for more sophisticated setups, if you need to change connection settings (e.g. when using django-environ):
#os.environ["DATABASE_URL"] = "postgres://myuser:mypassword@localhost:54324/mydb"
# Connect to Django ORM
django.setup()

# process data
# Review.objects.create(reviewText='MyAgency', predictSentiment='POSITIVE', actualSentiment='POSITIVE')
from tracker.models import Review
from tracker.models import PosScores
from tracker.models import NegScores
from tracker.models import WeightedAvg



def retrain(reviews, testDF):

    print(Fore.YELLOW + "Re-training model...")


    path = os.getcwd()
    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'

    tfidf_filename = ModelUtils.getLastModel(transformerPath)
    model_filename = ModelUtils.getLastModel(modelPath)

    print(tfidf_filename)
    print(model_filename)

    loaded_tfidf = pickle.load(open(tfidf_filename, 'rb'))
    loaded_model = pickle.load(open(model_filename, 'rb'))

    if not isinstance(reviews, pd.DataFrame):
        sys.exit("Not a pandas dataframe!!")

    if 'sentiment' not in reviews:
        sys.exit("Sentiment column not included in supplied DataFrame, please use dataprocess.addSenimentColumn()")

    print("DataFrame loaded")

    data = reviews['reviewText']

    ## model on new data
    X_new = loaded_tfidf.transform(data)
    y_new = reviews['sentiment']

    loaded_model.partial_fit(X_new,y_new,classes=np.unique(y_new))

    print("Model trained using the new data")

    path = os.getcwd()
    transformerPath = path + '\\savedModels\\transformer\\'
    modelPath = path + '\\savedModels\\model\\'

    # # new model saved to disk
    tfidf_filename = ModelUtils.getModelFileName(transformerPath)
    tfidf_filename = transformerPath + tfidf_filename
    pickle.dump(loaded_tfidf, open(tfidf_filename, 'wb'))
    print('New transformer saved to disk')
    #
    model_filename = ModelUtils.getModelFileName(modelPath)
    model_filename = modelPath + model_filename
    pickle.dump(loaded_model, open(model_filename, 'wb'))
    print(model_filename)
    print('New model saved to disk')

    print(proc.style.GREEN("Models trained and saved to disk") + proc.style.RESET(""))

    print(testDF.head())

    ## validate new model on the original test data
    test_set = testDF
    text_test, y_test = test_set['reviewText'].apply(lambda x: np.str_(x)), test_set['sentiment']

    # df['Review'].apply(lambda x: np.str_(x))

    ## model transform on test data
    X_test = loaded_tfidf.transform(text_test)


    print("SGD/LogReg Accuracy on new training set: {:.3f}".format(loaded_model.score(X_new, y_new)))
    print("SGD/LogReg Accuracy on original test set: {:.3f}".format(loaded_model.score(X_test, y_test)))

    path = os.getcwd()
    file = path + '\\savedModels\\accuracy\\modelAccuracy.csv'
    p = PosScores.objects.last()

    num  = p.id
    string = str(num)
    newFileName = 'retrained_model'+ string +'.sav'
    acc_score = loaded_model.score(X_new, y_new)
    acc_score = str(acc_score)

    newRow = string+','+newFileName+','+acc_score
    try:
        f1 = open(file, 'a', encoding='utf-8')
    except PermissionError as err:
        print(err)
        raise FileNotFoundError

    with open(file, 'a', encoding='utf-8') as f:
        f.write(newRow+'\n')

    f.close()

    return model_filename, tfidf_filename


# def test():
#     path = os.getcwd()
#     file = path + '/savedModels/accuracy/modelAccuracy.csv'
#     import csv
#     p = PosScores.objects.last()
#
#     num  = p.id +1
#     string = str(num)
#     newFileName = 'retrained_model'+ string +'.sav'
#     acc_score = loaded_model.score(X_new, y_new)
#     acc_score = str(acc_score)
#
#     newRow = string+','+newFileName+','+acc_score
#     with open(file, 'w') as f:
#         f.write(newRow+'\n')  # TRAILING NEWLINE
#
#     f.close()
# test()
    # with open(file,'w') as f:
    #