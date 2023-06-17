from src.cli.dataprocessing import preprocess
import pandas as pd
import numpy as np


def test_clean_variables():
    X = pd.DataFrame({
        'Age': [10, 52, np.nan, "?"],
        'Name': ['aurelien', '? ', ' c', " jack"]
    })

    # Define expected output after preprocessing
    X_clean = pd.DataFrame({
        'Age': [10, 52, np.nan, np.nan],
        'Name': ['aurelien', np.nan, 'c', 'jack']
    })

    preprocess.clean_variables(X)

    pd.testing.assert_frame_equal(X, X_clean)