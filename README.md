# Context
This repository contains material for Chronic Kidney Disease classification, using machine learning technics.
=> ADD DEFINITION OF CDK
This project is based using the dataset from the (UCI)[https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease] database.
The 2 goals are :
1. Determine risk factors for CKD
2. Determine potential CKD subtypes.

After a small research on about CKD, we can find that the main risk factors can be age, diabetes (related with glucose).


## Notebooks
The notebooks are used for Exploratory Data Analysis, to understand what's behind the dataset, and what are the available informations.
Details are the Notebook itself, but main take away points :
- Initial dataset is corrupted, with some format issues (missing column, trailing tabs).
- Data itself is not cleaned inside of the column, with cleaning to do both for numerical and categorical variables.
- There are many missing variables, that can and should be treated with some imputation.
- The target variable is a bit unbalanced.
First notebook (Preliminary_cleaning.ipynb) was used for cleaning the initial corrupted file
Second notebook (EDA.ipynb) was used for exploration of the data and of the variables, and make some trials.

## Main script and how to execute
Application has been developed with Python 3.9.13.
The script is a CLI executable, and necessary packages can be installed this way :
'''bash
pip install .
'''

Is you also want to run not only the source application, but also the exploratory notebooks, you can do the following way :
'''bash
pip install .[notebook]
'''


Once all setup,
'''bash
python -m src.cli --input "C:\Users\devil\Documents\Data_Science\CDK_project\01.RAW_DATA\Chronic_Kidney_Disease\chronic_kidney_disease_full.xlsx" -t "Class" -a "classification"
'''

CLI Helpers (can run "python -m src.cli -h" to see that)
'''
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path towards the input file in excel format
  --target-col TARGET_COL, -t TARGET_COL
                        Column containing the target column to predict
  --algorithm-task ALGORITHM_TASK, -a ALGORITHM_TASK
                        Type of task needed : possible options are classification or clustering. Any other string won't be accpted as input
'''

# Tests
In order to have a tested code, some functionalities of the code (like preprocessing) are checked with unit testing, to ensure that our functions are behaving as expected, in case we make any changes one day.
This tests can be launched the following way :

'''bash
TBD
'''

# 1. risk factors for CKD 
For this task, I have trained a Logistic Regression model for classification, on the original dataset with 70% used for training and 30% for evaluation. MIssing values have been imputed by the mean/mode, for numerical/categorical variables.
A Logistic Regression allows us to have interpretable results to understand the causes of the CKD, and measure the impact of each variable on the probability of heaving a CKD.

Classification Report:
              precision    recall  f1-score   support

           0       0.95      1.00      0.98        42
           1       1.00      0.97      0.99        78

    accuracy                           0.98       120

    accuracy                           0.98       120
   macro avg       0.98      0.99      0.98       120
weighted avg       0.98      0.98      0.98       120

Our model achieves an Accuracy of 0.98 for predicting the CKD.


# 2. Subtypes of CKD
For this task, I have used a clustering algorithm called K-means to group the people having a CKD and see if there were some groups in this population, and discover subtypes of CKD.
