import json
from django.shortcuts import render, get_object_or_404
import os
# from databaseQueries import getMonthlyRange
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


    def get_context_data(self, *args, **kwargs):
        context = super(dataView, self).get_context_data(*args, **kwargs)

        context['total_reviews'] = Review.objects.all().count()

        return context

def stuff():

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
            query = self.request.GET.get('q')
            all = stuff()
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
                context['datum'] = stuff()
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

    dirFiles = sortDirFiles()
    # print(dirFiles)
    # dirFiles.pop(0)

    # reviews = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    ids = PosScores.objects.values_list('id', flat=True).distinct()
    idCount = PosScores.objects.values_list('id', flat=True).distinct().count()
    print(idCount)
    print(ids)
    print(len(dirFiles))
    # dirFiles = ['Original File'] + dirFiles
    arr = []
    count = 0
    for id in ids:
        dict = {}

        if count+1 == idCount:
            break
        dict['id'] = id
        dict['file'] = dirFiles[count]
        count+=1
        arr.append(dict)

    for ar in arr:
        print(ar)
    # print(dict)
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


class PerformanceListView(ListView):
    model = Review
    template_name = 'tracker/performanceScores.html'

    def get_context_data(self, **kwargs):

        if self.request.user.is_authenticated:

            context = super(PerformanceListView, self).get_context_data(**kwargs)

            # all = Review.objects.all()
            # context['total_reviews'] = all.count()
            context['distinct_months'], context['distinct_month_count'] = getMonthlyRange()
            context['class_results'], context['distinct_id_count'] = getWeightedAvg()


            return context

#function based view - a detailed look at the data
def class_detail(request, pk):   #pass through the primary key to the view

    vals = stuff()
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
        # 'is_liked': is_liked,
        # 'total_likes': post.total_likes(),
        # 'comments': comments,
        # 'comment_form': comment_form
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

# class EventListView(ListView):
#     model = Review
#     template_name = 'tracker/home.html'
#
#     #
#     # def get_queryset(self):
#     #     toggle = self.request.session.get('toggleTime',None)
#     #
#     #     if toggle is None:
#     #         self.request.session['toggleTime'] = False
#     #         toggle = False
#     #     if toggle:
#     #         # original qs
#     #         qs = super().get_queryset()
#     #         # filter by a variable captured from url, for example
#     #         return qs.order_by('-resolution_date')
#     #     else:
#     #         return Event.objects.all()
#
#     def get_context_data(self, **kwargs):
#         username = None
#         # event = get_object_or_404(Event, id=self.kwargs['pk'])
#         # toggle = self.request.session.get('toggleTime2', None)
#
#         if self.request.user.is_authenticated:
#
#             # username = self.request.user.username
#             context = super(EventListView, self).get_context_data(**kwargs)
#             #
#             # if toggle is None:
#             #     self.request.session['toggleTime2'] = False
#             #     toggle = False
#             # if toggle:
#             #     # print("toggle")
#             #     context['values'] = Event.objects.all().filter(assigned_person__username=username)
#             # else:
#             #     # print("else")
#             #     context['values'] = Event.objects.all().filter(assigned_person__username=username).order_by('-resolution_date')
#
#             all = Review.objects.all()
#             context['total_reviews'] = all.count()
#             context['predict_positive'] = all.filter(predictSentiment='positive').count()
#             context['predict_negative'] = all.filter(predictSentiment='negative').count()
#             context['actual_positive'] = all.filter(actualSentiment='positive').count()
#             context['actual_negative'] = all.filter(actualSentiment='negative').count()
#             context['distinct_months'], context['distinct_month_count'] = getMonthlyRange()
#
#             # context['actual_positive'] = all.filter(actualSentiment='positive').count()
#             # context['actual_negative'] = all.filter(actualSentiment='negative').count()
#             return context


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


def about(request):
    return render(request, 'tracker/about.html')

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



#### Query functions ####

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

def getWeightedAvg():
    reviews = Review.objects.order_by('batch_date').values_list('avg_batch_no', flat=True).distinct()
    results = []
    for id in reviews:
        obj = WeightedAvg.objects.get(pk=id)
        results.append(obj)

    return results, reviews.count()

