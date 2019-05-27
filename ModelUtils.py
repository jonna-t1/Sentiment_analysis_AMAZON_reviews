import os
import re
import sys
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

def sortDirFiles(dirPath):
    dirFiles = sorted(os.listdir(dirPath))  # list of directory files

    # re.match(pattern, string, flags=0)
    numberOfFiles = len(dirFiles)
    numberOfDigits = len(str(numberOfFiles))
    if numberOfDigits > 2:
        sys.exit("Please enter less than 100 files")

    fileArray = [[], []]

    for i in dirFiles:
        # print(i)
        m = re.search("^.*?\([^\d]*(\d+)[^\d]*\).*$", i)
        if m:
            # sortedArray.append(i)
            strDig = m.groups()[0]

            if len(strDig) == 1:
                fileArray[0].append(i)

            if len(strDig) == 2:
                fileArray[1].append(i)
        else:
            sys.exit("Incorrect file naming convention. Uses (n) to determine file order - windows batch naming convention")

    file_list = [item for sublist in fileArray for item in sublist]
    print(file_list)
    return file_list

def getModelFileName(path):
    dirFiles = os.listdir(path)  # list of directory files
    lastFilename = dirFiles[-1]

    # guards against an empty directory
    p = PosScores.objects.last()
    num = p.id+1
    num = str(num)

    ##check if transformer or model
    lastFilename = lastFilename[:-4]    # removing extension
    result = ''.join([i for i in lastFilename if not i.isdigit()])  #removing digits

    if result == 'retrained_model':
        fileName = 'retrained_model' + num + '.sav'
    elif result == 'retrained_tfidftransformer':
        fileName = 'retrained_tfidftransformer' + num + '.sav'
    else:
        sys.exit("An error has occurred writing to a file")

    return fileName


# def sortingDirs(path):
#     fileName = ''
#     dirFiles = os.listdir(path)  # list of directory files
#
#     if dirFiles == []:
#         fileName = 'RocCurve1.png'
#         return fileName
#     else:
#         dirFiles.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#
#         lastFilename = dirFiles[-1]
#         file = lastFilename[:-4]
#         numbersInFile = [int(x) for x in file if x.isdigit()]
#         result = ''.join([i for i in file if not i.isdigit()])
#         #
#         numbersInFile = list(map(str, numbersInFile))
#         numbersInFile = ''.join([i for i in numbersInFile])
#         numbersInFile = int(numbersInFile)
#
#         count = numbersInFile + 1
#         fileNo = str(count)
#         fileName = result + fileNo + '.png'
#         return fileName


def getLastModel(path):
    dirFiles = os.listdir(path)  # list of directory files

    if dirFiles == []:
        sys.exit("Empty Directory, please include the inital model")

    lastFilename = dirFiles[-1]
    lastFilename = path + lastFilename
    return lastFilename


def getModel(path,num):
    dirFiles = os.listdir(path)  # list of directory files

    if dirFiles == []:
        sys.exit("Empty Directory, please include the inital model")

    lastFilename = dirFiles[num]
    lastFilename = path + lastFilename
    return lastFilename