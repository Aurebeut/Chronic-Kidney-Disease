import argparse
import pickle
import sys
from datetime import date

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from src.cli.dataprocessing import preprocess
from src.cli.models import classif, clustering

RANDOM_STATE = 7


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path towards the input file in excel format",
        type=str,
    )
    parser.add_argument(
        "--target-col",
        "-t",
        required=True,
        type=str,
        help="Column containing the target column to predict",
    )
    parser.add_argument(
        "--algorithm-task",
        "-a",
        required=True,
        type=str,
        help="Type of task needed : possible options are classification or clustering. Any other string won't be accpted as input",
    )
    args = parser.parse_args(argv)

    # read the file
    print("Loading the file ...")
    df = pd.read_excel(args.input)

    print("Preprocessing the file ...")
    # renaming columns for clarity purposes
    new_column_names = {
        "age": "Age",
        "bp": "BloodPressure",
        "sg": "SpecificGravity",
        "al": "Albumin",
        "su": "Sugar",
        "rbc": "RedBloodCells",
        "pc": "PusCell",
        "pcc": "PusCellClumps",
        "ba": "Bacteria",
        "bgr": "BloodGlucoseRandom",
        "bu": "BloodUrea",
        "sc": "SerumCreatinine",
        "sod": "Sodium",
        "pot": "Potassium",
        "hemo": "Hemoglobin",
        "pcv": "PackedCellVolume",
        "wc": "WhiteBloodCellCount",
        "rc": "RedBloodCellCount",
        "htn": "Hypertension",
        "dm": "DiabetesMellitus",
        "cad": "CoronaryArteryDisease",
        "appet": "Appetite",
        "pe": "PedalEdema",
        "ane": "Anemia",
        "class": "Class",
    }
    df = df.rename(columns=new_column_names)

    # defining the categorical/numerical variables
    # maybe automatically detect type of column?
    cols_numerical = [
        "Age",
        "BloodPressure",
        "SpecificGravity",
        "Albumin",
        "Sugar",
        "BloodGlucoseRandom",
        "BloodUrea",
        "SerumCreatinine",
        "Sodium",
        "Potassium",
        "Hemoglobin",
        "PackedCellVolume",
        "WhiteBloodCellCount",
        "RedBloodCellCount",
    ]
    cols_categorical = [
        "RedBloodCells",
        "PusCell",
        "PusCellClumps",
        "Bacteria",
        "Hypertension",
        "DiabetesMellitus",
        "CoronaryArteryDisease",
        "Appetite",
        "PedalEdema",
        "Anemia",
    ]
    cols_of_variables = [j for i in [cols_numerical, cols_categorical] for j in i]
    target_col = args.target_col

    # preprocess variables
    df[target_col] = [preprocess.clean_variables(x) for x in df[target_col]]
    for col in cols_of_variables:
        df[col] = [preprocess.clean_variables(x) for x in df[col]]
    df[target_col].replace(["ckd", "notckd"], [1, 0], inplace=True)
    y = df[target_col]
    X = df.drop([target_col, "Index_AD"], axis=1)

    # Split target variable and explaining variables
    if args.algorithm_task == "classification":
        # Impute the missing data
        print("Imputing missing data")
        for col in cols_categorical:
            preprocess.impute_mode(X, col)
        for col in cols_numerical:
            preprocess.impute_mean(X, col)

        # Shuffle and split the data before working on features and on model
        X, y = shuffle(X, y, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )

        # Scale/standardize before applying any model
        scaler = StandardScaler()
        X_train[cols_numerical] = scaler.fit_transform(X_train[cols_numerical])
        X_test[cols_numerical] = scaler.transform(X_test[cols_numerical])

        encoder = LabelEncoder()
        X_train[cols_categorical] = X_train[cols_categorical].apply(
            lambda col: encoder.fit_transform(col)
        )
        X_test[cols_categorical] = X_test[cols_categorical].apply(
            lambda col: encoder.fit_transform(col)
        )

        y_pred, model = classif.xgboost_classification(X_train, y_train, X_test)
        classif.model_evaluation(y_test, y_pred)

    elif args.algorithm_task == "clustering":
        # As we want to identify the sub types ok CKD, it's better focusing on CKD people only
        df_ckd = df[df[target_col] == 1]
        X = df_ckd.drop([target_col, "Index_AD"], axis=1)

        # Impute the missing data
        print("Imputing missing data")
        for col in cols_categorical:
            preprocess.impute_mode(X, col)
        for col in cols_numerical:
            preprocess.impute_mean(X, col)

        # Preprocess categorical features (one-hot encoding)
        encoder = LabelEncoder()
        X[cols_categorical] = X[cols_categorical].apply(
            lambda col: encoder.fit_transform(col)
        )

        # Helper to choose nb of clusters
        clustering.k_means_elbow_curve(X, RANDOM_STATE)

        nb_clusters = int(
            input(
                "Considering the curve, please choose the number of clusters (it must be an integer) based on the elbow method, and then press enter : "
            )
        )

        # Clustering part
        results_df_signif = clustering.k_means_clustering(
            X, cols_numerical, cols_categorical, nb_clusters, RANDOM_STATE, 0.01
        )
        print(results_df_signif)
    else:
        print(
            "Please provide a correct argument for algorithm_task, either 'classification' or 'clustering'"
        )


if __name__ == "__main__":
    main()
