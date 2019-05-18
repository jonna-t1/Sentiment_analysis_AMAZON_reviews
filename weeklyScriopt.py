# import datetime
#
# date = input("Enter date - %Y-%m-%d : ")
# aDate = datetime.datetime.strptime(date,"%Y-%m-%d")
#
# dates = []
#
# for i in range(52):
#     week = datetime.timedelta(weeks = i)
#     week = aDate + week
#     dates.append(week)
# def funct(batch, **kwargs):
#     print("stumep")
#
#
# def greet_me(**kwargs):
#     for key, value in kwargs.items():
#         print("{0} = {1}".format(key, value))
#         print(key)
import pandas as pd

testDF = pd.read_csv(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp\Datasets\TestData\TestData.csv')
print(testDF.head())

print(testDF['reviewText'])
# print(len(dates))
# print(dates)
import os
import sys


### assuming correct file indexing by the OS
# path = os.getcwd()
#
# dirFiles = os.listdir(path + '/savedModels/model/')  # list of directory files
# print(dirFiles)
#
# lastFilename = dirFiles[-1]
#
# vals = [int(x) for x in lastFilename if x.isdigit()]
#
# if vals == []:
#     print("save file finalised_model.sav1")
#     # return
#
# if len(vals) > 1:
#     print("An error has occured writing to model to file")
#     # return
#
# fileNo = vals[0]+1
# print("New file added: finalised_model.sav"+str(fileNo))




