from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score


def logistic_regression(X, y) :
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)
    # Get the coefficients for each feature
    feature_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': logistic_model.coef_[0]})
    # Print the feature coefficients
    print(feature_coefficients)
    return logistic_model

def xgboost_classification(X, y) :
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    return 


def model_evaluation(y_train, y_pred) :
    print("Classification Report:")
    print(classification_report(y_train, y_pred))
    print("Accuracy:", accuracy_score(y_train, y_pred))
    