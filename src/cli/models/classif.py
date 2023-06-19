import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def logistic_regression(X_train, y_train, X_test):
    """Performs logistic regression over a dataset, for classification.

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variables
        X_test (pd.DataFrame): test features

    Returns:
        y_pred (np.array): predicted variable
        logistic_model (sklearn.model): fitted model
    """
    logistic_model = LogisticRegression(penalty="l2")
    print(type(logistic_model))
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    # Get the coefficients for each feature
    feature_coefficients = pd.DataFrame(
        {"Feature": X_train.columns, "Coefficient": logistic_model.coef_[0]}
    )
    # Print the feature coefficients
    print(feature_coefficients)
    return y_pred, logistic_model


def xgboost_classification(X_train, y_train, X_test):
    """Performs xgboost classification over a dataset, for classification.

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target variables
        X_test (pd.DataFrame): test features

    Returns:
        y_pred (np.array): predicted variable
        logistic_model (sklearn.model): fitted model
    """
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    # Get feature importances
    feature_importances = xgb_model.feature_importances_
    selected_features = X_train.columns[feature_importances > 0.000000]

    # Subset dataset to selected features
    X_selected = X_train[selected_features]

    # Compute mean value or mode for each selected feature based on target variable
    result_df = pd.DataFrame(
        columns=["Feature", "Mean (class=1)", "Mean (class=0)", "P-Value"]
    )
    for feature in selected_features:
        mean_class_1 = X_selected.loc[y_train == 1, feature].mean()
        mean_class_0 = X_selected.loc[y_train == 0, feature].mean()

        # Perform t-test to check significance of difference
        _, p_value = ttest_ind(
            X_selected.loc[y_train == 1, feature], X_selected.loc[y_train == 0, feature]
        )

        result_df = result_df.append(
            {
                "Feature": feature,
                "Mean (class=1)": mean_class_1,
                "Mean (class=0)": mean_class_0,
                "P-Value": p_value,
            },
            ignore_index=True,
        )
    result_df_significant = result_df[result_df["P-Value"] < 0.05]
    print(result_df_significant)
    return y_pred, xgb_model


def model_evaluation(y_test, y_pred):
    """Returns classification metrics.

    Args:
        y_test (np.array): real target variable
        y_pred (np.array): predicted target variable

    Returns:
    """
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
