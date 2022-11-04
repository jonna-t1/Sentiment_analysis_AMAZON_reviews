from classifierFuncs.classifier import fetch_balance as fetch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from classifierFuncs.classifier import get_info
from classifierFuncs.algoConfig import batchAlgs, GS_top2, GS_SGD, saveModel
from sklearn.linear_model import LogisticRegression,SGDClassifier


#fetch requires the number of rows in the review database 
train_set, test_set = fetch(100000, 500)
text_test, y_test = test_set['reviewtext'], test_set['sentiment'] # test_set['overall'] provides labels
text_train = train_set['reviewtext']
y_train = train_set['sentiment']
## using the bag of words model
# TfidfVectorizer used over CountVectorizer as it not only focuses on the frequency of words present in the
# corpus but also provides the importance of the words
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3), stop_words='english')
tfidf = vectorizer.fit(text_train)
X_train = tfidf.transform(text_train)
X_test = tfidf.transform(text_test)
# get_info(tfidf) ## prints info on CountVectorizer() when fitted to the data
## Identifying rough machine learning algorithms that perform well - hyperparameters not tested
# batchAlgs(X_train, y_train, X_test, y_test) ## Comment out after initial run
## Testing hyperparameter tuning, using Grid search (GS)
# GS_top2(X_train, y_train, X_test, y_test)      # batchAlgs() identified LogReg and SVC performed best
# GS_SGD(X_train, y_train, X_test, y_test)        # Using SGDClassifier

# SGD was the best performing model with parameters:  {'loss': 'log_loss', 'penalty': 'elasticnet'}
sgd = SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=1000)
sgd.fit(X_train, y_train)
print("SGD/LogReg Accuracy on training set: {:.3f}".format(sgd.score(X_train, y_train)))
print("SGD/LogReg Accuracy on test set: {:.3f}".format(sgd.score(X_test, y_test)))

saveModel(tfidf,sgd)


 
