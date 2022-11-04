
import heapq
import os
import pickle

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import heatmap, visualize_coefficients
import matplotlib.pyplot as plt
import numpy as np


def batchAlgs(X_train, y_train, X_test, y_test):

    accuracy_dict={}

    lg = LogisticRegression(solver='lbfgs', max_iter=1000)
    lg.fit(X_train, y_train)
    print("Test set predictions: {}".format(lg.predict(X_test)))
    print("Logistic regression accuracy: {:.2f}".format(lg.score(X_test, y_test)))
    accuracy_dict["LogReg"] = lg.score(X_test, y_test)

    clf = KNeighborsClassifier(n_neighbors=9)
    clf.fit(X_train, y_train)
    print("Test set predictions: {}".format(clf.predict(X_test)))
    print("KNeighborsClassifier accuracy: {:.2f}".format(clf.score(X_test, y_test)))
    accuracy_dict["KNeighbors"] = clf.score(X_test, y_test)

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    print("Decision tree Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Decision tree Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    accuracy_dict["DTree"] = tree.score(X_test, y_test)

    ## random forest implementation
    forest = RandomForestClassifier(n_estimators=100, random_state=2)
    forest.fit(X_train, y_train)
    print("Random Forest Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Random Forest Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
    accuracy_dict["RForest"] = forest.score(X_test, y_test)

    # Gradient boosting
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)
    print("Gradient boosting Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Gradient boosting Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
    accuracy_dict["GBoost"] = gbrt.score(X_test, y_test)

    # Kernelized Support Vector Machines
    svm = SVC().fit(X_train, y_train)
    print("SVC Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
    print("SVC Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))
    accuracy_dict["SVM"] = svm.score(X_test, y_test)

    # Multi-layer perceptron
    mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
    print("MLPClassifier Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
    accuracy_dict["MLP1"] = mlp.score(X_test, y_test)

    # MLP, 10 hidden units
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    print("MLPClassifier 10 hidden layers Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
    print("MLPClassifier 10 hidden layers Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))
    accuracy_dict["MLP2"] = mlp.score(X_test, y_test)

    # selects the top 5 best performing algorithmns
    top_5 = heapq.nlargest(5, accuracy_dict, key=accuracy_dict.get)
    print("Best performing ML algorithms...")
    print(top_5)

def GS_top2(X_train, y_train, X_test, y_test):
    # ## logistic regression
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                  }
    
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation LogisticReg score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)

    # SVM/SVC - support vector machine
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation SVM score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Test data score: {:.2f}".format(grid.score(X_test, y_test))) # test data

def GS_SGD(X_train, y_train, X_test, y_test):
    # SVM/SVC - support vector machine
    param_grid = [
        # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'loss': ['hinge', 'log_loss'], 'penalty': ['l2', 'l1', 'elasticnet']}
    ]

    grid = GridSearchCV(SGDClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation score for SGD: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Test data score: {:.2f}".format(grid.score(X_test, y_test))) # test data

def ngramModel(text_train, y_train):
    pipe = make_pipeline(TfidfVectorizer(min_df=5), SGDClassifier())
    # running the grid search takes a long time because of the
    # relatively large grid and the inclusion of trigrams
    param_grid = {"sgdclassifier__loss": ['hinge', 'log_loss'], 
                "sgdclassifier__penalty": ['l2', 'l1', 'elasticnet'],
                # "logisticregression__"
                "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters:\n{}".format(grid.best_params_))

    # extract scores from grid_search
    scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
    # visualize heat map
    heatmp = heatmap(
        scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
        xticklabels=param_grid['sgdclassifier__loss'],
        yticklabels=param_grid['tfidfvectorizer__ngram_range'])
    plt.colorbar(heatmp)
    plt.savefig('./img/heatmap.png')
    plt.show()
    
    # extract feature names and coefficients
    vect = grid.best_estimator_.named_steps['tfidfvectorizer']
    feature_names = np.array(vect.get_feature_names())
    coef = grid.best_estimator_.named_steps['sgdclassifier'].coef_[0]
    visualize_coefficients(coef, feature_names, n_top_features=40)
    plt.savefig('./img/coef_visual.png')
    plt.show()

def saveModel(tfidf, sgd):
    """
    Saves the transformer and model to disk
    """
    path = os.getcwd()
    transformerPath = path + '/savedModels/transformer/'
    modelPath = path + '/savedModels/model/'
    # save the transformer to disk
    tfidf_filename = transformerPath+'finalised_tfidftransformer.sav'
    pickle.dump(tfidf, open(tfidf_filename, 'wb'))
    # save the model to disk
    model_filename = modelPath+'finalised_model.sav'
    pickle.dump(sgd, open(model_filename, 'wb'))

    try:
        os.path.isfile(tfidf_filename) 
        os.path.isfile(model_filename) 
        print(f"Transformer saved at {tfidf_filename}")
        print(f"Model saved at {model_filename}")
    except Exception as e:
        print(e)