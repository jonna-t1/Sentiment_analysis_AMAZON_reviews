import os
import re
import sys

import retrainModel as retrain
import pandas as pd
import pickle
import dataprocess as proc
from sklearn.metrics import classification_report
import ModelUtils as utils

def classify(reviews):

    path = os.getcwd()
    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'

    # load the model from disk
    tfidf_filename = utils.getLastModel(transformerPath)
    model_filename = utils.getLastModel(modelPath)
    loaded_tfidf = pickle.load(open(tfidf_filename, 'rb'))
    loaded_model = pickle.load(open(model_filename, 'rb'))

    ## loading dummy data
    data = reviews['reviewText']

    ## model prediction on new data
    X_new = loaded_tfidf.transform(data)

    y_pred = loaded_model.predict(X_new)
    print(loaded_model.predict_proba(X_new))

    target_names = ['positive', 'negative']
    y_true = reviews['sentiment']

    reviews['prediction'] = y_pred
    y_pred = reviews['prediction']
    print(reviews.head())
    print(y_true.count())
    print(y_pred.count())
    print(classification_report(y_true, y_pred, target_names=target_names))
    report = classification_report(y_true, y_pred, target_names=target_names)

    classArr = process_report(report)
    # classDF = pd.DataFrame(columns=['precision', 'recall', 'f1', 'support'])
    # print(classArr)
    # for i in range(5):
    #     classDF.loc[i] = classArr[i]

    return reviews, classArr

def process_report(report):
    report_data = []
    lines = report.split('\n')
    for line in lines:
        row = {}
        row_data = line.split('     ')
        report_data.append(row_data)

    report_dat = []
    report_data[2].pop(0)
    report_data[3].pop(0)
    report_data[5].pop(0)
    report_data[6].pop(0)
    report_data[7].pop(0)

    report_dat.append(report_data[2])
    report_dat.append(report_data[3])
    report_dat.append(report_data[5])
    report_dat.append(report_data[6])
    report_dat.append(report_data[7])

    # report_data = report_dat[len(report_dat) - 3:]
    report_data = report_dat

    extra = []
    for arr in report_data:
        if len(arr) == 4:
            continue
        extra = arr[2].split()
        # extra = re.split('\s+', arr[2])
        arr.pop(2)
        arr.append(extra)

    # good up to here

    for arr in report_data:
        if len(arr) == 4:
            continue
        if len(arr) == 3:
            for ar in arr[2]:
                arr.append(ar)
            arr.pop(2)

    # print(report_data)
    for arr in report_data:
        for i in arr:
            i.replace(" ", "")

    # print(report_data)

    return report_dat

