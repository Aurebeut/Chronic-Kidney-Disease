import re

import numpy as np
import pandas as pd


def clean_variables(var):
    """Preprocess to handle missing values, ?, nan and variables with trailing tabs/spaces.

    Args:
        var (str): variable to clean

    Returns:
        str: cleaned variable
    """
    var = str(var)
    if var is not None:
        var = var.strip()
        var = re.sub(r"\t", "", var)
        var = re.sub(r"\?", "", var)
        var = var.strip()
    else:
        var = np.nan
    if var == "":
        var = np.nan
    return var


def impute_mode(df, col_name):
    """Impute missing values by their mode. Used for categorical variables

    Args:
        df (pd.DataFrame): dataframe containing data to impute
        col_name (str): name of the column to clean

    Returns:
        df[col_name] (pd.Series): cleaned series
    """
    df[col_name].fillna(df[col_name].mode()[0], inplace=True)
    return df[col_name]


def impute_mean(df, col_name):
    """Impute missing values by their mean. Used for numerical variables

    Args:
        df (pd.DataFrame): dataframe containing data to impute
        col_name (str): name of the column to clean

    Returns:
        df[col_name] (pd.Series): cleaned series
    """
    df[col_name] = df[col_name].apply(pd.to_numeric)
    df[col_name].fillna(df[col_name].mean(), inplace=True)
    return df[col_name]
