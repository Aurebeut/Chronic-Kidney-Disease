import pandas as pd
import argparse
import sys



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
    args = parser.parse_args(argv)
    #read the file
    df=pd.read_excel(args.input)

    #defining the categorical/numerical variables

    #preprocess variables

    #Impute the missing data

    #Scale/standardize before applying any model

    #Train and run the models

    #Give results


