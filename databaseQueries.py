import datetime
import os
import re
import sys
import django

sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp')
sys.path.append(r'C:\\Users\j.turnbull\PycharmProjects\SentimentApp\SentimentApp')

sys.path.append(r'C:\Users\j.turnbull\PycharmProjects\SentimentApp') # add path to project root dir
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SentimentApp.settings')


# Connect to Django ORM
django.setup()

# process data
from tracker.models import Review
from tracker.models import PosScores
from tracker.models import NegScores
from tracker.models import WeightedAvg




def printLast():
    batch_no=None
    last_batch_no = Review.objects.order_by('-batch_no').first().batch_no
    if last_batch_no == 1:
        batch_no = last_batch_no + 1
    # print(last_batch_no)
    return batch_no


def removeAll():

    Review.objects.all().delete()
    PosScores.objects.all().delete()
    NegScores.objects.all().delete()
    WeightedAvg.objects.all().delete()

    print("All tables data deletion Complete")


def deleteBatch(number):
    Review.objects.filter(pos_batch_no=number).delete()
    if Review.objects.filter(pos_batch_no=number).first() == None:
        print("Batch deletion completed successfully")

def rankTop10():
    top_scores = (Review.objects.order_by('-id').values_list('id', flat=True).distinct())
    top_records = (Review.objects
                   .order_by('-id')
                   .filter(score__in=top_scores[:10]))
    print(top_records)

def check4BlankDatabase():
    if Review.objects.all() == None:
        print("Blank REVIEW table")
    else:
        print("POPULATED REVIEW TABLE")
    if PosScores.objects.all() == None:
        print("Blank SCORES table")
    else:
        print("POPULATED POSSCORES TABLE")
    if NegScores.objects.all() == None:
        print("Blank SCORES table")
    else:
        print("POPULATED POSSCORES TABLE")


def checkReview():
    print(Review.objects.all().first())


def showTop10():
    for i in range(10):
        objDate = Review.objects.all()[i].batch_date
        objDate = str(objDate.strftime("%B %d, %Y"))
        print(str(Review.objects.all()[i].pos_batch_no) + '  ' + objDate)


# print(t.strftime('%m/%d/%Y'))

def changeTimeStamp(batch_id, date):
    # print("hi")
    Review.objects.filter(pos_batch_no = batch_id).update(batch_date = date)

def getWeeklyArray():
    aDate = input("Please supply a start date in this format 'ddmmyyyy': ")
    cleaned = [x for x in aDate if x.isdigit()]
    aDate = "".join(str(x) for x in cleaned)    #list2string
    aDate = datetime.datetime.strptime(aDate, "%d%m%Y").date()  #string 2 datetime obj

    dates = []

    for i in range(52):
        week = datetime.timedelta(weeks = i)
        week = aDate + week
        dates.append(week)

    return dates

import datetime
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)

def getMonthlyArray():
    somedate = datetime.date.today()
    dates = []

    for i in range(12):
        date = add_months(somedate, i)
        dates.append(date)

    return dates

# q = Review.objects.values('avg_batch_no').all()
# q = Review.objects.values('avg_batch_no').distinct()
# # date_querySet = Review.objects.all().values_list('batch_date', flat=True)
# print(q)

def addMontlyDates():

    batchID_querySet = PosScores.objects.all().values_list('id', flat=True)
    valSplit = round(len(batchID_querySet) / 12)

    dates = getMonthlyArray()
    newDate = dates[0]
    count = 0
    for id in batchID_querySet:
        if id % valSplit+1 == 0:
            count += 1
            newDate = dates[count]
        changeTimeStamp(id, newDate)


def cleanModelsDir():
    dirPath = r'C:\Users\j.turnbull\PycharmProjects\SentimentApp\savedModels\model\\'
    print(dirPath)
    dirFiles = os.listdir(dirPath)  # list of directory files
    for file in dirFiles:
        if 'retrained_model.sav' in file and os.path.exists(dirPath + file):
            os.remove(dirPath + file)
            print(file + " Removed!")


def cleanTransformersDir():
    dirPath = r'C:\Users\j.turnbull\PycharmProjects\SentimentApp\savedModels\transformer\\'
    print(dirPath)
    dirFiles = os.listdir(dirPath)   # list of directory files
    for file in dirFiles:
        if 'retrained_tfidftransformer.sav' in file and os.path.exists(dirPath + file):
            os.remove(dirPath + file)
            print(file + " Removed!")


#########################
### execute script  #####
#########################

# print(getMonthlyArray())

# print(at)
# viewBatchIDs()
# monthlyChange()
# printLast()
# removeAll()
# # check4BlankDatabase()
# checkReview()
# checkScores()
# showTop10()

# t = datetime.datetime(2012, 2, 23, 0, 0)
# changeTimeStamp(1, t)
