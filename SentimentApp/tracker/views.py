import json
from django.shortcuts import render, get_object_or_404

from .forms import EventForm, RequestForm
from .models import Event, Review, Request
from django.http import HttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.mail import EmailMessage, send_mail
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView

from django.conf import settings


def home(request):
    context = {
        'events': Event.objects.all(),
        'values': Event.objects.all().filter(assigned_person__username='j.turnbull')


    }
    return render(request, 'tracker/home.html', context)

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


class EventListView(ListView):
    model = Event
    template_name = 'tracker/home.html'
    context_object_name = 'events'

    def get_queryset(self):
        toggle = self.request.session.get('toggleTime',None)

        if toggle is None:
            self.request.session['toggleTime'] = False
            toggle = False
        if toggle:
            # original qs
            qs = super().get_queryset()
            # filter by a variable captured from url, for example
            return qs.order_by('-resolution_date')
        else:
            return Event.objects.all()

    def get_context_data(self, **kwargs):
        username = None
        # event = get_object_or_404(Event, id=self.kwargs['pk'])
        toggle = self.request.session.get('toggleTime2', None)

        if self.request.user.is_authenticated:

            username = self.request.user.username
            context = super(EventListView, self).get_context_data(**kwargs)

            if toggle is None:
                self.request.session['toggleTime2'] = False
                toggle = False
            if toggle:
                # print("toggle")
                context['values'] = Event.objects.all().filter(assigned_person__username=username)
            else:
                # print("else")
                context['values'] = Event.objects.all().filter(assigned_person__username=username).order_by('-resolution_date')

            all = Review.objects.all()
            context['total_reviews'] = all.count()
            context['predict_positive'] = all.filter(predictSentiment='positive').count()
            context['predict_negative'] = all.filter(predictSentiment='negative').count()
            context['actual_positive'] = all.filter(actualSentiment='positive').count()
            context['actual_negative'] = all.filter(actualSentiment='negative').count()
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

class EventDetailView(DetailView):
    model = Event

class RequestDetailView(DetailView):
    model = Request

class RequestCreateView(CreateView):
    model = Request
    form_class = RequestForm

    login_url = '/login/'
    redirect_field_name = 'redirect_to'


class EventCreateView(LoginRequiredMixin, CreateView):
    model = Event
    form_class = EventForm

    login_url = '/login/'
    redirect_field_name = 'redirect_to'

    def form_valid(self, form):
        form.instance.assigned_person = self.request.user
        return super().form_valid(form)


class EventUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Event
    # form_class = EventForm

    fields = [
        'type',
        'reference',
        'status',
        'resolution_date',
        'priority',
        'assigned_team',
        'assigned_person',
        'summary'
    ]

    login_url = '/login/'
    redirect_field_name = 'redirect_to'

    # def form_valid(self, form):
    #     form.instance.assigned_person = self.request.user
    #     return super().form_valid(form)

    def test_func(self):
        event = self.get_object()
        if self.request.user == event.assigned_person:
            return True
        return False


class EventDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Event
    success_url = '/'

    def test_func(self):
        event = self.get_object()
        if self.request.user == event.assigned_person:
            return True
        return False


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

def send_email(request):
    # msg = EmailMessage('Request Callback',
    #     #                    'Here is the message.', to=['j.turnbull@accenture.com'])
    #     # msg.send()
    subject = "DATA BREACH!"
    from_email = settings.EMAIL_HOST_USER
    to_email = ["j.turnbull@accenture.com"]
    signup_message = "YOUR INCIDENT WILL EXPIRE IN 7 DAYS"
    send_mail(subject=subject, from_email=from_email, recipient_list=to_email, message=signup_message)
    return HttpResponse("<h1> Breach report sent! </h1>")