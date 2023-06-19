from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def logistic_regression(X_train, y_train, X_test) :
    """Performs logistic regression over a dataset, for classification.

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variables
        X_test (pd.DataFrame): test features

    Returns:
        y_pred (np.array): predicted variable
        logistic_model (sklearn.model): fitted model 
    """
    logistic_model = LogisticRegression(penalty='l2')
    print(type(logistic_model))
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    # Get the coefficients for each feature
    feature_coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': logistic_model.coef_[0]})
    # Print the feature coefficients
    print(feature_coefficients)
    return y_pred, logistic_model


def xgboost_classification(X_train, y_train, X_test) :
    """Performs xgboost classification over a dataset, for classification.

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variables
        X_test (pd.DataFrame): test features

    Returns:
        y_pred (np.array): predicted variable
        logistic_model (sklearn.model): fitted model 
    """
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    return y_pred, xgb_model


def model_evaluation(y_test, y_pred) :
    """Returns classification metrics.

    Args:
        y_test (np.array): real target variable
        y_pred (np.array): predicted target variable

    Returns:
    """
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))