from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import xgboost as xgb
from numpy import mean

import util

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477


# input: X_train, Y_train and X_test
# output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):
    # TODO: train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
    # use default params for the classifier
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    return model.predict(X_test)


# input: X_train, Y_train and X_test
# output: Y_pred
def svm_pred(X_train, Y_train, X_test):
    # TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
    # use default params for the classifier
    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    return model.predict(X_test)


# input: X_train, Y_train and X_test
# output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
    # TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
    # IMPORTANT: use max_depth as 5. Else your test cases might fail.
    model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    return model.predict(X_test)

def randomForest_pred(X_train, Y_train, X_test):
    # TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
    # IMPORTANT: use max_depth as 5. Else your test cases might fail.
    model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)

    return model.predict(X_test)

def xgb_pred(X_train, Y_train, X_test):
    #params = {"objective": "binary:logistic", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
    #          'max_depth': 5, 'alpha': 10}
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "learning_rate": 0.3,
              "reg_lambda": 10, "use_label_encoder": False,  # as we have done encoding
              "max_depth": 5, "subsample": 1}
    classification = xgb.XGBClassifier(**params)
    classification.fit(X_train, Y_train)

    return classification.predict(X_test)

# input: Y_pred,Y_true
# output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    # TODO: Calculate the above mentioned metrics
    # NOTE: It is important to provide the output in the same order
    acc = accuracy_score(Y_true, Y_pred)
    auc_ = roc_auc_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_true, Y_pred)

    return acc, auc_, precision, recall, f1score


# input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName, Y_pred, Y_true):
    print("______________________________________________")
    print("Classifier: " + classifierName)
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred, Y_true)
    print("Accuracy: " + str(acc))
    print("AUC: " + str(auc_))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1-score: " + str(f1score))
    print("______________________________________________")
    print("")

def run_models(df):
    #Split the dataset into X and Y
    Y = df['income']
    X = df.drop('income', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=RANDOM_STATE)
    #print("X_train.dtypes = ", X_train.dtypes)
    #print("Y_train.dtypes = ", Y_train.dtypes)
    display_metrics("Decision Tree", decisionTree_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("Logistic Regression", logistic_regression_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("SVM", svm_pred(X_train, Y_train, X_test), Y_test)
    display_metrics("XGBoost", xgb_pred(X_train, Y_train, X_test), Y_test)
