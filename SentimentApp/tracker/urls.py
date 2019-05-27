from django.conf.urls import url
from django.urls import path
from django.views.i18n import JavaScriptCatalog
from django.contrib.staticfiles.urls import static

from SentimentApp import settings
from .views import ReviewListView, RequestCreateView, RequestDetailView, PerformanceListView, PredictCountsListView, \
    AvgScoreDetailView, PosScoreDetailView, NegScoreDetailView, class_detail
from . import views

urlpatterns = [
    # path('', EventListView.as_view(), name='tracker-home'),
    path('', ReviewListView.as_view(), name='tracker-home'),
    path('performance/', PerformanceListView.as_view(), name='tracker-scores'),
    path('models/', views.trainedModelsView, name='models'),
    path('PredictCounts/', PredictCountsListView.as_view(), name='tracker-counts'),
    path('request/<int:pk>/', RequestDetailView.as_view(), name='request-detail'),
    path('avg/<int:pk>/', AvgScoreDetailView.as_view(), name='avg-detail'),
    path('positive/<int:pk>/', PosScoreDetailView.as_view(), name='pos-detail'),
    path('negative/<int:pk>/', NegScoreDetailView.as_view(), name='neg-detail'),
    path('request/new/', RequestCreateView.as_view(), name='request-create'),
    path('about/', views.about, name='tracker-about'),
    path('classification/<int:pk>/', views.class_detail, name='class-detail'),
    path('dataView/', views.dataView.as_view(), name='tracker-dataView'),
    path('dataView/matches/', views.matchView, name='tracker-match'),
    path('dataView/falsematches/', views.incorrectMatchView, name='tracker-falsematch'),
    path('posClassification/', views.posView.as_view(), name='tracker-posView'),
    path('negClassification/', views.negView.as_view(), name='tracker-negView'),
    path('avgClassification/', views.avgView.as_view(), name='tracker-avgView'),
    path('classification/', views.classificationView.as_view(), name='tracker-classTable'),
    path('get_time/', views.get_date, name='tracker-getEndTime'),
    path('toggleView/', views.toggleView, name='tracker-toggleView'),
    path('toggleView2/', views.toggleView2, name='tracker-toggleView2'),
    path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
]+ static(settings.STATIC_URL)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)