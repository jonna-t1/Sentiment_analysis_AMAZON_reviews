import json
import datetime
import pandas as pd
import calendar
from colorama import Fore
from django.core.paginator import Paginator
from django.db.models import F
import os
from .forms import RequestForm
from .models import Review, Request, PosScores, WeightedAvg, NegScores
from django.http import HttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.mail import EmailMessage, send_mail
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from pathlib import Path
import os
import re
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import FileUploadForm
from django.http import JsonResponse
from .ml_pipeline import train_update


def home(request):
    return render(request, 'tracker/home.html')

# def dataView(request):
#     get_object_or_404()
#     context = {
#         'datum' : Review.objects.all()
#     }
#     return render(request, 'tracker/data_view.html', context)

class dataView(ListView):
    model = Review
    template_name = 'tracker/data_view.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'datum'
    paginate_by = 10
    queryset = Review.objects.all()  # Default: Model.objects.all()

    def get_queryset(self):
        query = self.request.GET.get('q')
        if query:
            return Review.objects.filter(pos_batch_no=query)
        else:
            return Review.objects.all()

    def get_context_data(self, **kwargs):
        context = super(dataView, self).get_context_data(**kwargs)
        query = self.request.GET.get('q')
        if query:
            context['searchq'] = query

        return context

def filterIncorrect():
    falsePos = Review.objects.exclude(predictSentiment=F('actualSentiment'))
    return falsePos

def filterCorrect():
    matches = Review.objects.filter(predictSentiment=F('actualSentiment'))
    return matches


def incorrectMatchView(request):
    query = request.GET.get('q')
    if query:
        vals = filterIncorrect()
        vals = vals.filter(pos_batch_no=query)
    else:
        vals = filterIncorrect()
        query = ''

    paginator = Paginator(vals, 10)
    page = request.GET.get('page')
    vals = paginator.get_page(page)

    # context data
    context = {
        'datum': vals,
        'searchq': query,
    }

    return render(request, 'tracker/false_matches.html', context)


def matchView(request):

    query = request.GET.get('q')
    if query:
        vals = filterCorrect()
        vals = vals.filter(pos_batch_no=query)

    else:
        vals = filterCorrect()
        query = ''

    paginator = Paginator(vals, 10)
    page = request.GET.get('page')
    vals = paginator.get_page(page)
    # context data
    context = {
        'datum': vals,
        'searchq': query,
    }

    return render(request, 'tracker/matches.html', context)

def getScores():

    reviews = PosScores.objects.all().values_list('id', flat=True)
    results = []
    for id in reviews:#
        pos = PosScores.objects.get(pk=id)
        neg = NegScores.objects.get(pk=id)
        avg = WeightedAvg.objects.get(pk=id)
        results.append(pos)
        results.append(neg)
        results.append(avg)

    return results

class classificationView(ListView):
    model = PosScores
    template_name = 'tracker/classification_table.html'  # Default: <app_label>/<model_name>_list.html
    # paginate_by = 10

    def get_context_data(self, **kwargs):

        if self.request.user.is_authenticated:

            context = super(classificationView, self).get_context_data(**kwargs)
            query = self.request.GET.get('query')
            all = getScores()
            # reviews = Review.objects.order_by('batch_date').values_list('avg_batch_no', flat=True).distinct()
            results = []
            if query:
                num = int(query)
                for obj in all:
                    # print(type(obj.id))
                    if num == obj.id:
                        print(num)
                        results.append(obj)

                context['datum'] = results
                # return context

            else:
                context['datum'] = getScores()
                # return context

            return context

def sortDirFiles():
    p = Path(os.getcwd())
    dirPath = p.parent / ('savedModels/model/')
    dirFiles = os.listdir(dirPath)  # list of directory files
    # print(dirFiles)

    numberOfFiles = len(dirFiles)
    numberOfDigits = len(str(numberOfFiles))

    if numberOfFiles > 9:

        fileArray = [[], []]

        for i in dirFiles:
            # print(type(i))

            num = [int(char) for char in i if char.isdigit()]

            # num = str(num)
            # print(num)
            if len(num) == 1:
                fileArray[0].append(i)

            if len(num) == 2:
                fileArray[1].append(i)

        file_list = [item for sublist in fileArray for item in sublist]
        return file_list

    if numberOfFiles < 9:
        return dirFiles.sort()

def trainedModelsView(request):

    p = Path(os.getcwd())
    file = p.parent / ('savedModels/accuracy/modelAccuracy.csv')
    # dirFiles = os.listdir(dirPath)  # list of directory files
    arr = []

    with open(file, 'r') as f:
        rows = f.readlines()
    count = 1
    for row in rows:
        hash = {}
        if count == 1:
            vals = row.split(',')
            hash['id'] = vals[0]
            hash['file'] = vals[1]
            hash['score'] = vals[2]
            count+=1
            continue
        vals = row.split(',')
        print(vals)
        hash['id'] = vals[0]
        hash['file'] = vals[1]
        hash['score'] = vals[2]
        count += 1

        arr.append(hash)

    context = {
        'files': arr,
    }

    return render(request, 'tracker/trained_models.html', context)

class posView(ListView):
    model = PosScores
    template_name = 'tracker/positiveClassification.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'datum'
    paginate_by = 10

class negView(ListView):
    model = NegScores
    template_name = 'tracker/negativeClassification.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'negs'
    paginate_by = 10

class avgView(ListView):
    model = WeightedAvg
    template_name = 'tracker/avgClassification.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'avgs'
    paginate_by = 10

class AvgScoreDetailView(DetailView):
    model = WeightedAvg
    template_name = 'tracker/avg_detail.html'

class PosScoreDetailView(DetailView):
    model = PosScores
    template_name = 'tracker/pos_detail.html'


class NegScoreDetailView(DetailView):
    model = NegScores
    template_name = 'tracker/neg_detail.html'


class ReviewListView(ListView):
    model = Review
    template_name = 'tracker/home.html'

    def get_context_data(self, **kwargs):

        if self.request.user.is_authenticated:

            context = super(ReviewListView, self).get_context_data(**kwargs)

            all = Review.objects.all()
            context['total_reviews'] = all.count()
            context['predict_positive'] = all.filter(predictSentiment='positive').count()
            context['predict_negative'] = all.filter(predictSentiment='negative').count()
            context['actual_positive'] = all.filter(actualSentiment='positive').count()
            context['actual_negative'] = all.filter(actualSentiment='negative').count()
            context['distinct_months'], context['distinct_month_count'] = getMonthlyRange()

            # context['actual_positive'] = all.filter(actualSentiment='positive').count()
            # context['actual_negative'] = all.filter(actualSentiment='negative').count()
            return context


def chart_view(request):
    # Query all data from the PosScores model
    scores = WeightedAvg.objects.all().order_by('id')

    # Prepare data for the chart
    labels = [str(score.id) for score in scores]  # Use ID as labels (you can adjust this to your dates if available)
    precision = [float(score.precision) for score in scores]
    recall = [float(score.recall) for score in scores]
    f1 = [float(score.f1) for score in scores]

    context = {
        'labels': labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return render(request, 'tracker/chart.html', context)

def get_pos_scores(request):
    # Fetch all data from PosScores
    scores = PosScores.objects.all().values('id', 'precision', 'recall', 'f1', 'support')

    # Return as JSON
    return JsonResponse(list(scores), safe=False)

class PerformanceListView(ListView):
    model = Review
    template_name = 'tracker/performanceScores.html'

    def get_context_data(self, **kwargs):

        if self.request.user.is_authenticated:

            context = super(PerformanceListView, self).get_context_data(**kwargs)

            # all = Review.objects.all()
            # context['total_reviews'] = all.count()
            # context['distinct_months'], context['distinct_month_count'] = getMonthlyRange()
            # context['class_results'], context['distinct_id_count'] = getWeightedAvg()
            # context['pos_results'], context['distinct_id_count_pos'] = getPos()
            # context['neg_results'], context['distinct_id_count_neg'] = getNeg()

            context['distinct_months'] = getMonthLabels()

            pos, neg, avg, final = getLatestBatchPerMonth()

            context['class_results']= pos
            context['pos_results'] = neg
            context['neg_results']= avg
            context['class_results_count'] = len(pos)
            context['final'] = final

            # for i in pos:
            #     try:
            #         print(i.precision)
            #     except AttributeError as err:
            #         continue
            #
            # for i in neg:
            #     try:
            #         print(i.precision)
            #     except AttributeError as err:
            #         continue
            # for i in avg:
            #     try:
            #         print(i.precision)
            #     except AttributeError as err:
            #         continue

            return context

#function based view - a detailed look at the data
def class_detail(request, pk):   #pass through the primary key to the view

    vals = getScores()
    arr = []
    Dict = {'pos': 'positive', 'neg': 'negative', 'avg': 'WeightedAvg'}
    values = ['positive', 'negative', 'weighted average']
    count = 0
    for val in vals:
        if val.id == pk:
            # arr.append(values[count])
            arr.append(val)
            count += 1
    # post = get_object_or_404(Post, id=pk) #get post based on the post number
    # comments = Comment.objects.filter(post=post).order_by('-id') #ordering latest shown first

    #context data
    context = {
        'vals': vals,
        'array': arr,
        'fileNo': pk,
    }

    return render(request, 'tracker/classification_detail.html', context)

class PredictCountsListView(ListView):
    model = Review
    template_name = 'tracker/predict_counts.html'

    def get_context_data(self, **kwargs):

        if self.request.user.is_authenticated:

            context = super(PredictCountsListView, self).get_context_data(**kwargs)

            all = Review.objects.all()
            context['total_reviews'] = all.count()
            context['predict_positive'] = all.filter(predictSentiment='positive').count()
            context['predict_negative'] = all.filter(predictSentiment='negative').count()
            context['actual_positive'] = all.filter(actualSentiment='positive').count()
            context['actual_negative'] = all.filter(actualSentiment='negative').count()
            context['distinct_months'], context['distinct_month_count'] = getMonthlyRange()

            # context['actual_positive'] = all.filter(actualSentiment='positive').count()
            # context['actual_negative'] = all.filter(actualSentiment='negative').count()
            return context



def toggleView(request):
    toggle = request.session.get('toggleTime', None)

    data = {}
    status = 200

    if toggle is None:
        request.session['toggleTime']=True
    elif toggle:
        request.session['toggleTime']=False
    else:
        request.session['toggleTime'] = True

    return HttpResponse(json.dumps(data), content_type='application/json', status=status)

def toggleView2(request):
    toggle = request.session.get('toggleTime2', None)

    data = {}
    status = 200

    if toggle is None:
        request.session['toggleTime2']=True
    elif toggle:
        request.session['toggleTime2']=False
    else:
        request.session['toggleTime2'] = True

    return HttpResponse(json.dumps(data), content_type='application/json', status=status)



class RequestDetailView(DetailView):
    model = Request

    def get_context_data(self, **kwargs):
        context = super(RequestDetailView, self).get_context_data(**kwargs)
        context['form'] = RequestForm()
        return context

class RequestCreateView(CreateView):
    model = Request
    form_class = RequestForm

    login_url = '/login/'
    redirect_field_name = 'redirect_to'

def data_upload_page(request):
    return render(request, 'tracker/upload.html')


def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded file
            uploaded_file = form.cleaned_data['file']

            # Define the path to save the file in the DATA directory
            data_dir = os.path.join(settings.BASE_DIR, 'DATA')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)  # Create the DATA directory if it doesn't exist

            # Save the file to the DATA directory
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # return HttpResponse(f"File uploaded successfully to {file_path}.")
            return redirect('tracker-train')
    else:
        form = FileUploadForm()
    return render(request, 'tracker/upload.html', {'form': form})

def train_model(request):
    return render(request, 'tracker/train.html')

from django.http import JsonResponse
import time
def start_counting(request):
    train_update()
    # Simulate a long-running operation (e.g., counting to 100)
    # count = 0
    # while count < 100:
    #     time.sleep(0.1)  # Simulate some processing
    #     count += 1
    count = "done"

    return JsonResponse({'status': 'done', 'count': count})

def about(request):
    return render(request, 'tracker/about.html')
#### Query functions ####
def get_date(request):
    data = Event.objects.all()
    event_dates = []
    for d in data:
        ed = d.resolution_date
        ed = "{0}-{1}-{2} {3}:{4}:{5}".format(ed.year,ed.month,ed.day,ed.hour,ed.minute,ed.second)
        ev_id = [d.id,ed,]
        event_dates.append(ev_id)
        # print(json.dumps(event_dates))
    return HttpResponse(json.dumps(event_dates), content_type="application/json")


def getMonthlyRange():

    # reviews = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    reviews = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()

    print(reviews.first())
    print(reviews.last())
    print(reviews.count())
    vals = []
    for i in reviews:
        vals.append(i)

    return vals, reviews.count()

#
# def subtract_one_month(t):
#     one_day = datetime.timedelta(days=1)
#     one_month_earlier = t - one_day
#     while one_month_earlier.month == t.month or one_month_earlier.day > t.day:
#         one_month_earlier -= one_day
#     return one_month_earlier


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
    # start = subtract_one_month(start)

    new = start.strftime('%Y-%m-%d')
    finish = months.last().strftime('%Y-%m-%d')

    dateLabels = pd.date_range(new, finish,
                  freq='MS').strftime("%Y-%b").tolist()
    # dateLabels = [months.first().strftime("%Y-%b")]+dateLabels

    return dateLabels



def getLatestBatchPerMonth():

    pos = []
    neg = []
    avg = []

    dates = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    first = dates.first()
    last = dates.last()
    print(last)

    try:
        first = first.strftime('%Y-%m-%d')
        finish = last.strftime('%Y-%m-%d')
    except AttributeError as err:
        first = datetime.date.today()
        finish = datetime.date.today()


        print(err)

    # first = subtract_one_month(first)
    # first = first.strftime('%Y-%m-%d')
    # finish = last.strftime('%Y-%m-%d')
    dateRange = pd.date_range(first, finish,
                               freq='MS').tolist()
    hash = {}
    count = 0
    for date in dateRange:

        # print(type(date.month))
        # sys.exit("stop")
        hash[date.month] = date.year
        count+=1


    final = {}
    for key, value in hash.items():
        # print("{}  {}".format(key, value))
        month = key
        year = value

        arr = calendar.monthrange(year, month)
        start_date = datetime.date(year, month, 1)
        end_date = datetime.date(year, month, arr[1])
        vals = Review.objects.filter(batch_date__range=(start_date, end_date)).values_list('pos_batch_no', flat=True)
        lastReview = Review.objects.filter(batch_date__range=(start_date, end_date)).last()

        if lastReview:
            lastDate = lastReview.batch_date
        else:
            continue

        lastDate = lastDate.strftime("%Y-%b")

        lastVal = vals.last()
        pos_score = ""
        neg_score = ""
        avg_score = ""

        if not lastVal:
            pos_score = PosScores.objects.none()  # assign empty if there arent any batches in that month
            neg_score = NegScores.objects.none()
            avg_score = WeightedAvg.objects.none()
            continue

        try:
            pos_score = PosScores.objects.get(pk=lastVal)
        except PosScores.DoesNotExist:
            pos_score = PosScores.objects.none()  # assign empty if there arent any batches in that month

        try:
            neg_score = NegScores.objects.get(pk=lastVal)
        except NegScores.DoesNotExist:
            neg_score = NegScores.objects.none()
        try:
            avg_score = WeightedAvg.objects.get(pk=lastVal)
        except WeightedAvg.DoesNotExist:
            avg_score = WeightedAvg.objects.none()

        print(avg_score)

        hash ={}

        #append the score obj
        pos.append(pos_score)
        neg.append(neg_score)
        avg.append(avg_score)

        hash['positive'] = pos_score
        hash['negative'] = neg_score
        hash['wAverage'] = avg_score

        final[lastDate] = hash
        # print(lastDate)

    return pos, neg, avg, final

one, two, three, final = getLatestBatchPerMonth()

# for key, value in final.items():
#     print(key+":   {}".format(value))

print(final)