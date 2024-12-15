from django.urls import path
from django.views.i18n import JavaScriptCatalog
from django.contrib.staticfiles.urls import static

from SentimentApp import settings
from .views import ReviewListView, RequestCreateView, RequestDetailView, PerformanceListView, PredictCountsListView, \
    class_detail
from . import views

urlpatterns = [
    path('', ReviewListView.as_view(), name='tracker-home'),
    path('performance/', PerformanceListView.as_view(), name='tracker-scores'),
    path('models/', views.trainedModelsView, name='models'),
    path('PredictCounts/', PredictCountsListView.as_view(), name='tracker-counts'),
    path('request/<int:pk>/', RequestDetailView.as_view(), name='request-detail'),
    path('request/new/', RequestCreateView.as_view(), name='request-create'),
    path('classification/<int:pk>/', views.class_detail, name='class-detail'),
    path('dataView/', views.dataView.as_view(), name='tracker-dataView'),
    path('dataView/matches/', views.matchView, name='tracker-match'),
    path('dataView/falsematches/', views.incorrectMatchView, name='tracker-falsematch'),
    path('classification/', views.classificationView.as_view(), name='tracker-classTable'),
    path('upload/', views.upload_file, name='upload_file'),
    path('train/', views.train_model, name='tracker-train'),
    path('start-counting/', views.start_counting, name='start_counting'),
    # path('chart/', views.score_view, name='tracker-chart'),
    path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
    path('chart/', views.chart_view, name='chart'),
    path('get-pos-scores/', views.get_pos_scores, name='get_pos_scores'),
]+ static(settings.STATIC_URL)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

