import re
import spacy
import mglearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt


precision = make_scorer(precision_score, pos_label='positive')
recall = make_scorer(recall_score, pos_label='positive')
f1 = make_scorer(f1_score, pos_label='positive')


def ImplementLDA(text_train):
    vect = CountVectorizer(max_features=10000, max_df=.15)
    X = vect.fit_transform(text_train)

    lda = LatentDirichletAllocation(n_topics=10, learning_method="batch", max_iter=25, random_state=0)
    # We build the model and transform the data in one step
    # Computing transform takes some time,
    # and we can save time by doing both at once
    document_topics = lda.fit_transform(X)

    print("lda.components_.shape: {}".format(lda.components_.shape))

    # for each topic (a row in the components_), sort the features (ascending).
    # Invert rows with [:, ::-1] to make sorting descending
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    # get the feature names from the vectorizer:
    feature_names = np.array(vect.get_feature_names())

    # for each topic (a row in the components_), sort the features (ascending).
    # Invert rows with [:, ::-1] to make sorting descending
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    # get the feature names from the vectorizer:
    feature_names = np.array(vect.get_feature_names())

    # Print out the 10 topics:
    mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                               sorting=sorting, topics_per_chunk=5, n_words=10)


    lda100 = LatentDirichletAllocation(n_topics=100, learning_method="batch",
                                       max_iter=25, random_state=0)
    document_topics100 = lda100.fit_transform(X)

    topics = np.array([7, 16, 24, 25, 28, 36, 37, 41, 45, 51, 53, 54, 63, 89, 97])

    sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
                               sorting=sorting, topics_per_chunk=5, n_words=20)


    top = np.argsort(document_topics100[:, 54])[::-1]
    # print the five documents where the topic is most important

    for i in top[:10]:
        senten = text_train.iloc[i]
        print(senten+"\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    topic_names = ["{:>2} ".format(i) + " ".join(words)
                   for i, words in enumerate(feature_names[sorting[:, :2]])]
    # two column bar chart:
    for col in [0, 1]:
        start = col * 50
        end = (col + 1) * 50
        ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
        ax[col].set_yticks(np.arange(50))
        ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
        ax[col].invert_yaxis()
        ax[col].set_xlim(0, 300)
        yax = ax[col].get_yaxis()
        yax.set_tick_params(pad=130)
    plt.tight_layout()
    plt.show()

def customerVectorizer(text_train, y_train):
    # regexp used in CountVectorizer:
    regexp = re.compile('(?u)\\b\\w\\w+\\b')
    # load spacy language model
    en_nlp = spacy.load('en', disable=['parser', 'ner'])
    old_tokenizer = en_nlp.tokenizer
    # replace the tokenizer with the preceding regexp
    en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
        regexp.findall(string))

    # create a custom tokenizer using the SpaCy document processing pipeline
    # (now using our own tokenizer)
    def custom_tokenizer(document):
        doc_spacy = en_nlp(document)
        return [token.lemma_ for token in doc_spacy]

    # define a count vectorizer with the custom tokenizer
    lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
    # transform text_train using CountVectorizer with lemmatization
    X_train_lemma = lemma_vect.fit_transform(text_train)
    print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

    # standard CountVectorizer for reference
    vect = CountVectorizer(min_df=5).fit(text_train)
    X_train = vect.transform(text_train)
    print("X_train.shape: {}".format(X_train.shape))
    X_train_lemma.shape: (25000, 21637)
    X_train.shape: (25000, 27271)
    # build a grid-search using only 1% of the data as training set:
    from sklearn.model_selection import StratifiedShuffleSplit

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99,
                                train_size=0.01, random_state=0)
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
    # perform grid search with standard CountVectorizer
    grid.fit(X_train, y_train)
    print("Best cross-validation score "
          "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
    # perform grid search with Lemmatization
    grid.fit(X_train_lemma, y_train)
    print("Best cross-validation score "
          "(lemmatization): {:.3f}".format(grid.best_score_))


def ngramModel(text_train, y_train):
    pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
    # running the grid search takes a long time because of the
    # relatively large grid and the inclusion of trigrams
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
                  # "logisticregression__"
                  "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters:\n{}".format(grid.best_params_))

    # extract scores from grid_search
    scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
    # visualize heat map
    heatmap = mglearn.tools.heatmap(
        scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
        xticklabels=param_grid['logisticregression__C'],
        yticklabels=param_grid['tfidfvectorizer__ngram_range'])
    plt.colorbar(heatmap)
    plt.show()
    #
    # extract feature names and coefficients
    vect = grid.best_estimator_.named_steps['tfidfvectorizer']
    feature_names = np.array(vect.get_feature_names())
    coef = grid.best_estimator_.named_steps['logisticregression'].coef_[0]
    mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
    plt.show()

    # find 3-gram features
    mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
    # visualize only 3-gram features
    mglearn.tools.visualize_coefficients(coef.ravel()[mask],
                                         feature_names[mask], n_top_features=40)
    plt.show()

def trigramModel(text_train, y_train):
    pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
    # running the grid search takes a long time because of the
    # large grid and the inclusion of trigrams
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
    # "logisticregression__solver": ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters:\n{}".format(grid.best_params_))

def tdidfResults(text_train, y_train):
    pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None), LogisticRegression())
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, np.ravel(y_train,order='C'))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    # extracting the TfidfVectorizer from the pipeline:
    vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
    # transform the training dataset
    X_train = vectorizer.transform(text_train)
    # find maximum value for each of the features over the dataset
    max_value = X_train.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    # get feature names
    feature_names = np.array(vectorizer.get_feature_names())
    print("Features with lowest tfidf:\n{}".format(
    feature_names[sorted_by_tfidf[:20]]))
    print("Features with highest tfidf: \n{}".format(
    feature_names[sorted_by_tfidf[-20:]]))

    # The inverse document frequency values found on the training set are stored in the idf_ attribute:
    sorted_by_idf = np.argsort(vectorizer.idf_)
    print("Features with lowest idf:\n{}".format(
    feature_names[sorted_by_idf[:100]]))

    mglearn.tools.visualize_coefficients(
    grid.best_estimator_.named_steps["logisticregression"].coef_,
    feature_names, n_top_features=40)
    plt.show()
    
def stopWordRemoval(text_train, y_train):
    vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
    X_train = vect.transform(text_train)
    print("X_train with stop words:\n{}".format(repr(X_train)))

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                  }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

def top2(X_train, y_train, X_test, y_test):

    print("--- Logistic regression ---")
    accuracy_scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    precision_scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5, scoring=precision)   # with precision
    recall_scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5, scoring=recall)   # with recall
    f1_scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5, scoring=f1)   # with f1

    print("Mean cross-validation LogisticRegression accuracy: {:.2f}".format(np.mean(accuracy_scores)))
    print("Mean cross-validation LogisticRegression precision: {:.2f}".format(np.mean(precision_scores)))
    print("Mean cross-validation LogisticRegression recall: {:.2f}".format(np.mean(recall_scores)))
    print("Mean cross-validation LogisticRegression f1: {:.2f}".format(np.mean(f1_scores)))

    print("--- RandomForestClassifier ---")
    accuracy_scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5)
    precision_scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring=precision)   # with precision
    recall_scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring=recall)   # with recall
    f1_scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring=f1)   # with f1

    print("Mean cross-validation RandomForestClassifier accuracy: {:.2f}".format(np.mean(accuracy_scores)))
    print("Mean cross-validation RandomForestClassifier precision: {:.2f}".format(np.mean(precision_scores)))
    print("Mean cross-validation RandomForestClassifier recall: {:.2f}".format(np.mean(recall_scores)))
    print("Mean cross-validation RandomForestClassifier f1: {:.2f}".format(np.mean(f1_scores)))

def top2Tuning(X_train, y_train, X_test, y_test):

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                  }

    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation LogisticReg score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)

    # test data
    print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))

    n_estimators = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}
    grid = GridSearchCV(RandomForestClassifier(random_state=2), n_estimators, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation RandomForest score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)

    # test data
    print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))


def top4(X_train, y_train, X_test, y_test):

    # ## logistic regression
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
    #               'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    #               }
    #
    # grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    # # grid = GridSearchCV(LinearSVC(), param_grid, cv=5)
    # grid.fit(X_train, y_train)
    # print("Best cross-validation LogisticReg score: {:.2f}".format(grid.best_score_))
    # print("Best parameters: ", grid.best_params_)
    #
    # # test data
    # print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))
    #
    # # random forest
    # n_estimators = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}
    # grid = GridSearchCV(RandomForestClassifier(random_state=2), n_estimators, cv=5)
    # grid.fit(X_train, y_train)
    # print("Best cross-validation RandomForest score: {:.2f}".format(grid.best_score_))
    # print("Best parameters: ", grid.best_params_)
    #
    # # test data
    # print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))
    #
    # # gradient boosting
    # learning_rate = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
    # grid = GridSearchCV(GradientBoostingClassifier(random_state=2), learning_rate, cv=5)
    # grid.fit(X_train, y_train)
    # print("Best cross-validation GradientBoostingClassifier score: {:.2f}".format(grid.best_score_))
    # print("Best parameters: ", grid.best_params_)
    #
    # # test data
    # print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))

    # SVC - support vector machines
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation SVM score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)

    # test data
    print("Test data score: {:.2f}".format(grid.score(X_test, y_test)))


def batchAlgs(X_train, y_train, X_test, y_test):

    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    print("Test set predictions: {}".format(lg.predict(X_test)))
    print("Test set accuracy: {:.2f}".format(lg.score(X_test, y_test)))

    clf = KNeighborsClassifier(n_neighbors=9)
    clf.fit(X_train, y_train)
    print("Test set predictions: {}".format(clf.predict(X_test)))
    print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
    print("break\n")

    #####

    # # best results found at 3 or 9
    # training_accuracy = []
    # test_accuracy = []
    # # try n_neighbors from 1 to 10
    # neighbors_settings = range(1, 11)
    # for n_neighbors in neighbors_settings:
    #     # build the model
    #     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    #     clf.fit(X_train, y_train)
    #     # record training set accuracy
    #     training_accuracy.append(clf.score(X_train, y_train))
    #     # record generalization accuracy
    #     test_accuracy.append(clf.score(X_test, y_test))
    #     print("Train set accuracy: {:.2f} - {}".format(clf.score(X_train, y_train), n_neighbors))
    #     print("Test set accuracy: {:.2f} - {}".format(clf.score(X_test, y_test), n_neighbors))

    # plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    # plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("n_neighbors")
    # plt.legend()
    # plt.show()

    ##############

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    print("Decision tree Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Decision tree Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    print("DecisionTreeClassifier Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("DecisionTreeClassifier Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

    ## random forest implementation
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=100, random_state=2)
    forest.fit(X_train, y_train)

    print("Random Forest Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Random Forest Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

    # gradient boosting implementation
    from sklearn.ensemble import GradientBoostingClassifier

    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)
    print("Gradient boosting Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Gradient boosting Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

    ## To reduce overfitting, we could either apply stronger
    #  pre-pruning by limiting the maximum depth or lower the learning rate:

    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    print("Gradient boosting Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Gradient boosting Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train, y_train)
    print("Gradient boosting Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Gradient boosting Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

    # Kernelized Support Vector Machines
    from sklearn.svm import SVC

    # svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X_train, y_train)
    svm = SVC().fit(X_train, y_train)
    print("SVC Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
    print("SVC Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))

    svc = SVC(C=1000)
    svc.fit(X_train, y_train)
    print("SVC C=1000 Accuracy on training set: {:.3f}".format(
        svc.score(X_train, y_train)))
    print("SVC C=1000 Accuracy on test set: {:.3f}".format(svc.score(X_test, y_test)))

    ##neural nets
    # default is 100 hidden units
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
    print("MLPClassifier Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

    # 10 hidden units
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    print("MLPClassifier 10 hidden layers Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier 10 hidden layers Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

    # finding ideal vals
    # for n_hidden_nodes in [10, 100]:
    #     for alpha in [0.0001, 0.01, 0.1, 1]:
    #         mlp = MLPClassifier(solver='lbfgs', random_state=0,
    #                             hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
    #                             alpha=alpha)
    #         mlp.fit(X_train, y_train)
    #         print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)), )
    #         print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)

    print("MLPClassifier random state=42 Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier random state = 42 Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

    # try on rescaled features in tf-idf

    ## increase the alpha parameter (quite aggressively, from 0.0001 to 1) to add stronger
    # regularization of the weights

    mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
    mlp.fit(X_train, y_train)
    print("MLPClassifier aplha paramter 1 Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier alpha parameter 1 Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))


# def uncertaintyClassifiers():
#     ## uncertainty estimates from classifiers
#
#
#     # build the gradient boosting model
#     gbrt = GradientBoostingClassifier(random_state=0)
#     gbrt.fit(X_train, y_train)
#
#     print("X_test.shape:", X_test.shape)
#     print("Decision function shape:", gbrt.decision_function(X_test).shape)
#
#     print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))
#
#     print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test) > 0))
#     print("Predictions:\n{}".format(gbrt.predict(X_test)))
#
#     # make the boolean True/False into 0 and 1
#     greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
#     # use 0 and 1 as indices into classes_
#     pred = gbrt.classes_[greater_zero]
#     # pred is the same as the output of gbrt.predict
#     print("pred is equal to predictions: {}".format(
#         np.all(pred == gbrt.predict(X_test))))
#
#     decision_function = gbrt.decision_function(X_test)
#     print("Decision function minimum: {:.2f} maximum: {:.2f}".format(np.min(decision_function), np.max(decision_function)))
#
#     ## predicting probabilities
#     print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
#
#     # show the first few entries of predict_proba
#     print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))