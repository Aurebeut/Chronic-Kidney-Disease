from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score


def logistic_regression(X_train, y_train, X_test) :
    logistic_model = LogisticRegression()
    print("fitting model")
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    # Get the coefficients for each feature
    feature_coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': logistic_model.coef_[0]})
    # Print the feature coefficients
    print(feature_coefficients)
    return y_pred, logistic_model

def xgboost_classification(X, y) :
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    return y_pred, xgb_model


def model_evaluation(y_test, y_pred) :
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))