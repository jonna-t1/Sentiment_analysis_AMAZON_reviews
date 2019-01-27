from django.conf.urls import url
from django.urls import path
from django.views.i18n import JavaScriptCatalog

from .views import EventListView, EventDetailView, EventCreateView, EventUpdateView, EventDeleteView
from . import views

urlpatterns = [
    path('', EventListView.as_view(), name='tracker-home'),
    path('event/<int:pk>/', EventDetailView.as_view(), name='event-detail'),
    path('event/<int:pk>/update/', EventUpdateView.as_view(), name='event-update'),
    path('event/<int:pk>/delete/', EventDeleteView.as_view(), name='event-delete'),
    path('event/new/', EventCreateView.as_view(), name='event-create'),
    path('about/', views.about, name='tracker-about'),
    path('get_time/', views.get_date, name='tracker-getEndTime'),
    path('send_email/', views.send_email, name='tracker-sendEmail'),
    path('toggleView/', views.toggleView, name='tracker-toggleView'),
    path('toggleView2/', views.toggleView2, name='tracker-toggleView2'),
    path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
]
