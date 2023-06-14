import re
import numpy as np

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
        var = re.sub(r"\t","", var)
        var = re.sub(r"\?","", var)
        var = var.strip()
    else:
        var = np.nan
    if var == "":
        var = np.nan
    return var
