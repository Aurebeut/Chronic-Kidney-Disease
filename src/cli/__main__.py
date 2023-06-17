import pandas as pd
import argparse
import sys
from src.cli.dataprocessing import preprocess
from src.cli.models import classif
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
        help='Column containing the target column to predict',
    )
    parser.add_argument(
        "--algorithm-task",
        "-a",
        required=True,
        type=str,
        help='Type of task needed : possible options are classification or clustering. Any other string won\'t be accpted as input',
    )
    args = parser.parse_args(argv)

    #read the file
    print("Loading the file ...")
    df=pd.read_excel(args.input)

    print("Preprocessing the file ...")
    #renaming columns for clarity purposes
    new_column_names = {
        'age': 'Age',
        'bp': 'BloodPressure',
        'sg': 'SpecificGravity',
        'al': 'Albumin',
        'su': 'Sugar',
        'rbc': 'RedBloodCells',
        'pc': 'PusCell',
        'pcc': 'PusCellClumps',
        'ba': 'Bacteria',
        'bgr': 'BloodGlucoseRandom',
        'bu': 'BloodUrea',
        'sc': 'SerumCreatinine',
        'sod': 'Sodium',
        'pot': 'Potassium',
        'hemo': 'Hemoglobin',
        'pcv': 'PackedCellVolume',
        'wc': 'WhiteBloodCellCount',
        'rc': 'RedBloodCellCount',
        'htn': 'Hypertension',
        'dm': 'DiabetesMellitus',
        'cad': 'CoronaryArteryDisease',
        'appet': 'Appetite',
        'pe': 'PedalEdema',
        'ane': 'Anemia',
        'class': 'Class'
    }
    df = df.rename(columns=new_column_names)

    #defining the categorical/numerical variables
    #maybe automatically detect type of column?
    cols_numerical = ["Age", "BloodPressure", "SpecificGravity", "Albumin", "Sugar", "BloodGlucoseRandom", "BloodUrea", "SerumCreatinine", "Sodium", "Potassium", "Hemoglobin", "PackedCellVolume", "WhiteBloodCellCount", "RedBloodCellCount"]
    cols_categorical = ["RedBloodCells", "PusCell", "PusCellClumps", "Bacteria", "Hypertension", "DiabetesMellitus", "CoronaryArteryDisease", "Appetite", "PedalEdema", "Anemia"]
    cols_of_variables = [j for i in [cols_numerical, cols_categorical] for j in i]
    target_col = args.target_col

    #preprocess variables
    df[target_col] = [preprocess.clean_variables(x) for x in df[target_col]]
    for col in cols_of_variables :
        df[col] = [preprocess.clean_variables(x) for x in df[col]]
    df[target_col].replace(["ckd","notckd"],[1,0], inplace=True)

    #Split target variable and explaining variables
    X = df.drop([target_col,"Index_AD"], axis=1)
    y = df[target_col]

    #Impute the missing data
    print("Imputing missing data")
    for col in cols_categorical:
        preprocess.impute_mode(X, col)
    for col in cols_numerical:
        preprocess.impute_mean(X, col)

    #Scale/standardize before applying any model
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = RANDOM_STATE)

    print("Training the models ...")
    #Display feature importance
    if args.algorithm_task == "classification":
        print("Classification")
        y_pred, model = classif.logistic_regression(X,y)
        #y_pred, model_result = classif.xgboost_classification(X,y)
        #Save the model to a pickle file
        print(y_train)
        print(y_pred)

        classif.model_evaluation(y_train,y_pred)
    #Give results and the "best" model selected
    if args.algorithm_task == "clustering":
        print("Clustering")

if __name__ == "__main__":
    main()