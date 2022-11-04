import datetime
import os
import re
import sys
import django
import datetime
import calendar

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

# removeAll()


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


somedate = datetime.date.today()
d = datetime.timedelta(days=30)
t = somedate + d
# changeTimeStamp(27, t)

# def getWeeklyArray():
#     aDate = input("Please supply a start date in this format 'ddmmyyyy': ")
#     cleaned = [x for x in aDate if x.isdigit()]
#     aDate = "".join(str(x) for x in cleaned)    #list2string
#     aDate = datetime.datetime.strptime(aDate, "%d%m%Y").date()  #string 2 datetime obj
#
#     dates = []
#
#     for i in range(52):
#         week = datetime.timedelta(weeks = i)
#         week = aDate + week
#         dates.append(week)
#
#     return dates


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)



def getMonthlyArray():
    # somedate = datetime.date.today()
    somedate = datetime.datetime.strptime(aDate, "%d%m%Y").date()  #string 2 datetime obj


    aDate = input("Please supply a start date in this format 'ddmmyyyy': ")
    cleaned = [x for x in aDate if x.isdigit()]
    aDate = "".join(str(x) for x in cleaned)    #list2string
    somedate = datetime.datetime.strptime(aDate, "%d%m%Y").date()  #string 2 datetime obj

    dates = []

    for i in range(12):
        date = add_months(somedate, i)
        dates.append(date)

    return dates


def changeMonthlyDates():

    # batchID_querySet = PosScores.objects.all().values_list('id', flat=True)
    allReviews = Review.objects.all()
    review_batches = Review.objects.values('avg_batch_no').distinct().values_list('avg_batch_no', flat=True)
    dateTimeSet = Review.objects.values('avg_batch_no').distinct().values_list('batch_date', flat=True)

    print(dateTimeSet.first())
    print(dateTimeSet.last())
    # sys.exit("stop")

    print("Pre date change...")
    print(dateTimeSet)
    valSplit = 0

    dates = getMonthlyArray()
    print(dates)
    # sys.exit("stop")
    print(review_batches)

    ## changing timestamps
    count = 0
    for id in review_batches:
        print(id)
        changeTimeStamp(id, dates[count])
        count += 1

    # somedate = datetime.date.today()    # date of upload - 18/05/2019 change this to suit
    # somedate = somedate - timedelta(days=1)
    # somArr = []
    #
    # for i in dateTimeSet:
    #     if i.date() == somedate:
    #         somArr.append(i)

    print("Post change...")

    print(dateTimeSet.first())  #should be the provided date
    print(dateTimeSet.last())   # might be the upload date

    # for i in somArr:
    #     print("1")
        # Review.objects.filter(batch_date=i).delete()


# changeMonthlyDates()

# print(Review.objects.none())



def getMonths():
    dates = Review.objects.all().values_list('batch_date', flat=True)
    distinct = dates.distinct()
    # print(len(distinct))
    # sys.exit()

    somedate = datetime.date.today()    # date of upload - 18/05/2019 change this to suit
    vals = []
    # print(len(dates))

    # resetting times
    # for date in dates:
    #     val = resetTime(date)
    #     Review.objects.filter(batch_date=date).update(batch_date=val)


    for date in distinct:
        print(date.month)

        # if date.date() == somedate:
        #     vals.append(date)
            # date.remove()
        # print(date.month)
            # print(count += 1)

    # for val in vals:
    #     print(Review.objects.filter(batch_date = val))      # .delete to delete the outliers

# print(printLast())

# getMonths()


# print(Review.objects.all().first())
# print(Review.objects.all().last())

# poss = PosScores.objects.all().last()
# neg = NegScores.objects.all().last()
# avg = WeightedAvg.objects.all().last()
#
# somedate = datetime.date.today()
#
# event_1 = Review(reviewText = 'Problem', predictSentiment = 'positive', actualSentiment='positive',
#                  batch_date = somedate, pos_batch_no = poss, neg_batch_no = neg, avg_batch_no = avg)
# event_1.save()


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


def subtract_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month later.

    Note that the resultant day of the month might change if the following
    month has fewer days:

        >>> subtract_one_month(datetime.date(2010, 3, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_earlier = t - one_day
    while one_month_earlier.month == t.month or one_month_earlier.day > t.day:
        one_month_earlier -= one_day
    return one_month_earlier


def getMonthLabels():
    import pandas as pd

    # reviews = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    months = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()

    final = []
    array = []
    for month in months:
        mon = month.strftime("%Y-%b")
        array.append(mon)

    start = months.first()
    start = subtract_one_month(start)

    new = start.strftime('%Y-%m-%d')
    finish = months.last().strftime('%Y-%m-%d')

    dateLabels = pd.date_range(new, finish,
                  freq='MS').strftime("%Y-%b").tolist()
    dateLabels = [months.first().strftime("%Y-%b")]+dateLabels

    count = 0
    for date in dateLabels:
        if date == array[count]:
            final.append(date)
            count+=1
            continue
        if date != array[count] and date !=array[0]:
            final.append("")

    # for date in dateLabels:
    #     Review.objects.filter(batch_date=dat)

    print(final)

# getMonthLabels()

def getLatestBatchPerMonth():
    import pandas as pd
    import calendar

    pos = []
    neg = []
    avg = []

    dates = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    first = dates.first()
    last = dates.last()

    # first = subtract_one_month(first)
    first = first.strftime('%Y-%m-%d')
    finish = last.strftime('%Y-%m-%d')
    dateRange = pd.date_range(first, finish,
                               freq='MS').tolist()
    hash = {}
    count = 0
    for date in dateRange:

        # print(type(date.month))
        # sys.exit("stop")
        hash[date.month] = date.year
        count+=1

    # print(hash[0])
    for key, value in hash.items():
        # print("{}  {}".format(key, value))
        month = key
        # nextMonth = key + 1
        year = value
        arr = calendar.monthrange(year, month)
        start_date = datetime.date(year, month, 1)
        end_date = datetime.date(year, month, arr[1])
        vals = Review.objects.filter(batch_date__range=(start_date, end_date)).values_list('pos_batch_no', flat=True)

        lastVal = vals.last()
        # skip if there arent any batches in that month
        pos_score = ""
        neg_score = ""
        avg_score = ""

        try:
            pos_score = PosScores.objects.get(pk=lastVal)
        except PosScores.DoesNotExist:
            go = None
        try:
            neg_score = NegScores.objects.get(pk=lastVal)
        except NegScores.DoesNotExist:
            go = None
        try:
            avg_score = WeightedAvg.objects.get(pk=lastVal)
        except WeightedAvg.DoesNotExist:
            go = None

        pos.append(pos_score)
        neg.append(neg_score)
        avg.append(avg_score)

    return pos, neg, avg


# # one, two, three = getLatestBatchPerMonth()
#
# for i in one:
#     try:
#         print(i.precision)
#     except AttributeError as err:
#         continue
#
# for i in two:
#     try:
#         print(i.precision)
#     except AttributeError as err:
#         continue
# for i in three:
#     try:
#         print(i.precision)
#     except AttributeError as err:
#         continue


def getWeightAvg():
    reviews = Review.objects.order_by('batch_date').values_list('avg_batch_no', flat=True).distinct()
    for id in reviews:
        obj = WeightedAvg.objects.get(pk=id)
        print(obj)


# getWeightAvg()

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


def eachROC():
    path = os.getcwd()
    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'


    pos = PosScores.objects.all()
    firstID = PosScores.objects.first().id
    count = 1

    ## get from db
    for p in pos:

        if p.id == firstID:
            tfidf_filename = transformerPath + 'finalised_tfidftransformer.sav'
            model_filename = modelPath + 'finalised_model.sav'

        # load the model from disk
        tfidf_filename = utils.getModel(transformerPath, count)
        model_filename = utils.getModel(modelPath, count)
        f1 = open(tfidf_filename, 'rb')
        f2 = open(model_filename, 'rb')
        loaded_tfidf = pickle.load(f1)
        loaded_model = pickle.load(f2)

        reviews = pd.DataFrame()
        qs = Review.objects.filter(pos_batch_no=p.id)

        reviews = read_frame(qs)

        ## loading dummy data
        data = reviews['reviewText']

        ## model prediction on new data
        X_new = loaded_tfidf.transform(data)
        y_pred = loaded_model.predict(X_new)

        target_names = ['positive', 'negative']
        y_true = reviews['actualSentiment']

        probs = []
        probs = loaded_model.predict_proba(X_new)
        print(probs)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_true, probs)

        print('AUC: %.3f' % auc)
        # sys.exit("step")

        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label='positive')
        # plot no skill
        pyplot.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        pyplot.plot(fpr, tpr, marker='.')
        # show the plot
        # pyplot.show()
        ROCFileName1 = path + '\\savedModels\\ROC\\' + 'RocCurve' + str(p.id) + '.png'
        ROCFileName2 = path + '\\SentimentApp\\tracker\\static\\roc\\' + 'RocCurve' + str(p.id) + '.png'

        pyplot.savefig(ROCFileName1)
        pyplot.savefig(ROCFileName2)
        pyplot.close()

        count += 1
        f1.close()
        f2.close()

# eachROC()

def filterIncorrect():
    from django.db.models import F
    all = Review.objects.all()
    matches = Review.objects.filter(predictSentiment=F('actualSentiment'))
    falsePos = Review.objects.exclude(predictSentiment=F('actualSentiment'))

    for val in falsePos:
        print(val.predictSentiment + "  "+ val.actualSentiment)

    return matches, falsePos


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
