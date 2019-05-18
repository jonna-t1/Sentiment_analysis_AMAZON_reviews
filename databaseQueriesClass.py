import datetime
import os
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



class DBQuery:

    # self.

    def printLast(self):
        batch_no=None
        last_batch_no = Review.objects.order_by('-batch_no').first().batch_no
        if last_batch_no == 1:
            batch_no = last_batch_no + 1
        # print(last_batch_no)
        return batch_no


    def removeAll(self):

        Review.objects.all().delete()
        PosScores.objects.all().delete()
        NegScores.objects.all().delete()
        WeightedAvg.objects.all().delete()

        print("All tables data deletion Complete")


    def deleteBatch(number):
        Review.objects.filter(pos_batch_no=number).delete()
        if Review.objects.filter(pos_batch_no=number).first() == None:
            print("Batch deletion completed successfully")

    def rankTop10(self):
        top_scores = (Review.objects.order_by('-id').values_list('id', flat=True).distinct())
        top_records = (Review.objects
                       .order_by('-id')
                       .filter(score__in=top_scores[:10]))
        print(top_records)

    def check4BlankDatabase(self):
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


    def checkReview(self):
        print(Review.objects.all().first())


    def showTop10(self):
        for i in range(10):
            objDate = Review.objects.all()[i].batch_date
            objDate = str(objDate.strftime("%B %d, %Y"))
            print(str(Review.objects.all()[i].pos_batch_no) + '  ' + objDate)


    # print(t.strftime('%m/%d/%Y'))

    def changeTimeStamp(batch_id, date):
        # print("hi")
        Review.objects.filter(pos_batch_no = batch_id).update(batch_date = date)

# at = PosScores.objects.last()
# print(at)

# printLast()
# removeAll()
# # check4BlankDatabase()
checkReview()
# checkScores()
# showTop10()

# t = datetime.datetime(2012, 2, 23, 0, 0)
# changeTimeStamp(1, t)
