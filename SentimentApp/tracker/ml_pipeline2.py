import os
import ml_pipeline.data_extraction as extract
import ml_pipeline.pre_processing as pre_proc
import ml_pipeline.feature_extraction as features
import ml_pipeline.model_training as model_training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
from models import Review
from pathlib import Path

# Data extraction
# C:\Users\jturn\PycharmProjects\Sentiment_analysis_AMAZON_reviews\SentimentApp\DATA
path = Path(__file__).resolve().parent.parent / "DATA"
print(path)
# path = os.path.dirname(os.getcwd())+'\\DATA\\'
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

result = pd.DataFrame({"reviewTextRaw": X_test_raw, "ProcessedReview": X_test, "sentiment": y_test})

# result['reviewTextRaw']
first_10_reviews = Review.objects.all()[:10]

for review in first_10_reviews:
    print(review)

# X_train_raw = X_train_raw.rename(columns={"reviewText": "reviewTextRaw"})
# Concatenate along columns (indices match)
# result = pd.concat([raw_df, X_train], axis=1)
# print(result.tail(3))

# transformer = features.load_transformer()
# new_X = transformer.transform(X_train)
# X_test = transformer.transform(X_test)
#
# model = model_training.load_model()
# model.partial_fit(new_X, y_train, classes=np.unique(y_train))
#
# y_pred = model.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test_raw, y_pred)}")

