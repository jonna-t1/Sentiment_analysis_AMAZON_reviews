import os
import tracker.ml_pipeline_functions.data_extraction as extract
import tracker.ml_pipeline_functions.pre_processing as pre_proc
import tracker.ml_pipeline_functions.feature_extraction as features
# import tracker.ml_pipeline_functions.model_training as model_training
import tracker.ml_pipeline_functions.model_training as model_training
import tracker.db_interactions as db
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
from .models import Review, WeightedAvg
from pathlib import Path
from django.utils import timezone


# Data extraction
def train_update():
    # C:\Users\jturn\PycharmProjects\Sentiment_analysis_AMAZON_reviews\SentimentApp\DATA
    path = Path(__file__).resolve().parent.parent / "DATA"
    print(path)
    most_recent_file = extract.get_most_recent_file(path)

    raw_df = extract.getDF(most_recent_file)
    print(raw_df['reviewText'].tail(3))

    raw_df['sentiment'] = raw_df['overall'].apply(pre_proc.get_sentiment)
    raw_df['reviewText'] = raw_df['reviewText'].fillna("")
    labels = raw_df['sentiment']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(raw_df['reviewText'], labels, test_size=0.2, random_state=42)

    # Pre_processing
    df = pd.DataFrame(raw_df['reviewText'].apply(pre_proc.preprocess_text))
    df['sentiment'] = labels
    print(df.tail(3))

    X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['sentiment'], test_size=0.2, random_state=42)

    print(type(X_train))
    print(type(X_train_raw))

    result = pd.DataFrame({"reviewTextRaw": X_test_raw, "ProcessedReview": X_test, "actualSentiment": y_test})

    transformer = features.load_transformer()
    new_X = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    model_training.load_model()
    model_training.train(new_X, y_train)
    predicted_values, classification_report = model_training.predict(X_test, y_test)
    result["predictSentiment"] = predicted_values

    weighted_average = classification_report['weighted avg']
    poss_score = classification_report["1"]
    neg_score = classification_report["0"]
    ## Must be called before ReviewCreate()
    db.WeightedAvgCreate(weighted_average)
    db.PosScoresCreate(poss_score)
    db.NegScoresCreate(neg_score)
    # create review
    db.ReviewCreate(result)

    #
    # result["predictedValues"] = result["predictedValues"].replace(1,"positive")
    # result["predictedValues"] = result["predictedValues"].replace(0, "negative")
    # result["ActualSentiment"] = result["ActualSentiment"].replace(1,"positive")
    # result["ActualSentiment"] = result["ActualSentiment"].replace(0, "negative")
    # print(result.tail(3))
    #
    # # List to hold model instances
    # instances = []
    # # Loop through the DataFrame and create model instances
    # for index, row in result.iterrows():
    #     instance = Review(
    #         reviewText=row['reviewTextRaw'],
    #         predictSentiment=row['predictedValues'],
    #         actualSentiment=row['ActualSentiment']
    #     )
    #     instances.append(instance)
    #
    # # Bulk create all instances
    # Review.objects.bulk_create(instances)
    # Review.objects.filter(batch_date__gte=timezone.now().date()).delete()

    # first_10_reviews = Review.objects.all()[:10]
    # for review in first_10_reviews:
    #     print(review)
    # print column names
    # for field in Review._meta.fields:
    #     print(field.name)
    # print(result.columns)



