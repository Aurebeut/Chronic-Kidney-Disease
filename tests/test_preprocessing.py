from src.cli.dataprocessing import preprocess
import pandas as pd
import numpy as np


# def test_clean_variables():
#     X = pd.DataFrame(
#         {"Age": [10, 52, np.nan, "?"], "Name": ["aurelien", "? ", " c", " jack"]}
#     )

#     # Define expected output after preprocessing
#     X_clean = pd.DataFrame(
#         {"Age": [10.0, 52.0, np.nan, np.nan], "Name": ["aurelien", np.nan, "c", "jack"]}
#     )

#     for col in X.columns:
#         X[col] = [preprocess.clean_variables(x) for x in X[col]]

#     pd.testing.assert_frame_equal(preprocess.clean_variables(X), X_clean)


def test_clean_variables():
    # Create a sample dataframe
    data = {"col1": ["10", "?", " 10", "\t100"], "col2": ["  5  ", " ", "3 ", "\t"]}
    df = pd.DataFrame(data)

    # Apply clean_variables function to each column
    cols_of_variables = ["col1", "col2"]
    for col in cols_of_variables:
        df[col] = [preprocess.clean_variables(x) for x in df[col]]

    # Assert the expected results
    assert df["col1"].tolist() == ["10", np.nan, "10", "100"]
    assert df["col2"].tolist() == ["5", np.nan, "3", np.nan]
