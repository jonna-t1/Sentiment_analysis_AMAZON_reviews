
import os
from tkinter import Tk, filedialog
import pandas as pd
import pickle
import sys
import numpy as np
import dataprocess as proc
from colorama import Fore
import ModelUtils


def retrain(reviews, testDF, *args):

    if args != 'batch' or args != 'Batch':
        answer = input(Fore.YELLOW + "Are you sure you would like to retrain the model??? If you wish to continue press Y or y: ")
    elif args == 'batch' or args == 'Batch':
        answer = 'y'

    if answer != 'Y' and answer != 'y':
        exit('Retraining ABORTED')

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

    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'

    # # new model saved to disk
    tfidf_filename = ModelUtils.getModelFileName(transformerPath)
    pickle.dump(loaded_tfidf, open(tfidf_filename, 'wb'))
    print('New transformer saved to disk')
    #
    model_filename = ModelUtils.getModelFileName(modelPath)
    pickle.dump(loaded_model, open(model_filename, 'wb'))
    print('New model saved to disk')

    print(proc.style.GREEN("Models trained and saved to disk") + proc.style.RESET(""))

    ## validate new model on the original test data
    test_set = testDF
    text_test, y_test = test_set['reviewText'], test_set['sentiment'] # test_set['overall'] provides labels

    ## loading dummy data
    data = text_test['reviewText']
    ## model prediction on new data
    X_test = loaded_tfidf.transform(data)

    print("SGD/LogReg Accuracy on new training set: {:.3f}".format(loaded_model.score(X_new, y_new)))
    print("SGD/LogReg Accuracy on original test set: {:.3f}".format(loaded_model.score(X_test, y_test)))

