import os
import ml_pipeline.data_extraction as extract
import ml_pipeline.pre_processing as pre_proc
import ml_pipeline.feature_extraction as features
import ml_pipeline.model_training as model_training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Data extraction
path = os.path.dirname(os.getcwd())+'\\DATA\\'
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

# Feature extraction
df = features.extract(df)
print(df.tail(3))

print(df.columns)
X_train, X_test, y_train, y_test = train_test_split(df['combined_features'], df['sentiment'], test_size=0.2, random_state=42)

# print(X_train)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss', penalty='l2')
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
import numpy as np
X_train = np.array(X_train.tolist())  # Ensure a proper 2D array
# sgd.fit(X_train, y_train)
# model_training.load_model()
model_training.train(X_train, y_train)