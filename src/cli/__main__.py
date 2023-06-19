import pandas as pd
import argparse
import sys
from src.cli.dataprocessing import preprocess
from src.cli.models import classif
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import date
from sklearn.cluster import KMeans

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
    parser.add_argument(
        "--save-model",
        "-s",
        action="store_true",
        default=None,
        help="Flag to save your model in a pickle file, stored at the root of the repository.",
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

    # Split target variable and explaining variables
    if args.algorithm_task == "classification":
        X = df.drop([target_col, "Index_AD"], axis=1)
        y = df[target_col]

        # Impute the missing data
        print("Imputing missing data")
        for col in cols_categorical:
            preprocess.impute_mode(X, col)
        for col in cols_numerical:
            preprocess.impute_mean(X, col)

        # Scale/standardize before applying any model
        X, y = shuffle(X, y, random_state=RANDOM_STATE)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        print(type(X_train))
        print(type(y_train))
        # need to
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
        print("Classification")
        y_pred_logistic, model = classif.logistic_regression(X_train, y_train, X_test)
        classif.model_evaluation(y_test, y_pred_logistic)
        # Save the model to a pickle file

        if args.save_model:
            filename = f"{args.algorithm_task}_model.pkl"
            pickle.dump(model, open(filename, "wb"))
            with open(f"{args.algorithm_task}_{date.today()}.pkl", "wb") as file:
                pickle.dump(model, file)

    # Give results and the "best" model selected
    elif args.algorithm_task == "clustering":
        print("Clustering")
        # As we want to identify the sub types ok CKD, we need to focus on CKD people only
        df_ckd = df[df[target_col] == 1]
        X = df_ckd.drop([target_col, "Index_AD"], axis=1)

        # Impute the missing data
        print("Imputing missing data")
        for col in cols_categorical:
            preprocess.impute_mode(X, col)
        for col in cols_numerical:
            preprocess.impute_mean(X, col)

        # Preprocess numerical features (scale)
        scaler = StandardScaler()
        X[cols_numerical] = scaler.fit_transform(X[cols_numerical])
        # scaled_numerical_features = scaler.fit_transform(df_ckd[])

        # Preprocess categorical features (one-hot encoding)
        encoder = LabelEncoder()
        X[cols_categorical] = X[cols_categorical].apply(
            lambda col: encoder.fit_transform(col)
        )

        # Combine the scaled numerical features and encoded categorical features
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=6, n_init="auto")
        kmeans.fit(X)

        # Get the cluster labels
        cluster_labels = kmeans.labels_
        cluster_labels_df = pd.DataFrame({"Cluster Labels": cluster_labels})

        # Concatenate the df with the cluster labels
        new_df = pd.concat([X, cluster_labels_df], axis=1)

        # Compute the mean value of each column based on the cluster labels
        # new_df[cols_numerical] = scaler.inverse_transform(X[cols_numerical])
        # mean_values = new_df.groupby("Cluster Labels").mean()
        # print(mean_values)

    else:
        print(
            "Please provide a correct argument for algorithm_task, either 'classification' or 'clustering'"
        )


if __name__ == "__main__":
    main()
