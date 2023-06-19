# Context
This repository contains material for Chronic Kidney Disease classification, using machine learning technics. 
Chronic kidney disease (CKD) is a long-term condition characterized by a gradual loss of kidney function over time
This project is based using the dataset from the [UCI](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) database.
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
```bash
pip install .
```

Is you also want to run not only the source application, but also the exploratory notebooks, you can do the following way :
```bash
pip install .[notebook]
```


Once all setup,
```bash
python -m src.cli --input "C:\Users\devil\Documents\Data_Science\CDK_project\01.RAW_DATA\Chronic_Kidney_Disease\chronic_kidney_disease_full.xlsx" -t "Class" -a "classification"
```

To see CLI Helpers do the following :
```bash
python -m src.cli -h
```

# Tests
In order to have a tested code, some functionalities of the code (like preprocessing) are checked with unit testing, to ensure that our functions are behaving as expected, in case we make any changes one day.
This tests can be launched the following way :

```bash
python -m pytest
```

# 1. risk factors for CKD 
For this task, I have trained a XGBoost model for classification, on the original dataset with 70% used for training and 30% for evaluation. MIssing values have been imputed by the mean/mode, for numerical/categorical variables.
CLI used for reproducibility :
```bash
python -m src.cli --input "C:\Users\devil\Documents\Data_Science\CDK_project\01.RAW_DATA\Chronic_Kidney_Disease\chronic_kidney_disease_full.xlsx" -t "Class" -a "classification"
```
An XGBoost classifier allows us to define the most important features to understand the causes of the CKD.



Here are the features importances :
                Feature  Mean (class=1)  Mean (class=0)       P-Value
0                   Age        0.190294       -0.303060  4.906974e-05
1         BloodPressure        0.267364       -0.425801  6.981745e-09
2       SpecificGravity       -0.561771        0.894672  4.670483e-44
3               Albumin        0.487857       -0.776958  1.327538e-30
4    BloodGlucoseRandom        0.313847       -0.499831  5.930627e-12
5             BloodUrea        0.283322       -0.451217  7.228765e-10
6       SerumCreatinine        0.308416       -0.491181  1.460528e-11
7                Sodium       -0.352575        0.561508  5.097756e-15
8            Hemoglobin       -0.577095        0.919077  1.537435e-47
9      PackedCellVolume       -0.544022        0.866405  2.303023e-40
10  WhiteBloodCellCount        0.135916       -0.216458  3.993840e-03
11    RedBloodCellCount       -0.459832        0.732325  1.337632e-26
12         Hypertension        0.587209        0.000000  3.139888e-28


We can deduce that :
- Older people have more chances to get CKD.
- A high blood pressure (and hypertension) / Albumin / Blood Glucose / Blood Urea / SerumCreatinine  is correlated with CKD.
- A low Specific Gravity / Glucose in blood / Sodium in blood / Hemoglobin / is correlated with CKD.
- A low volume of Packed cells and Red Blood Cells is correlated with CKD.
- A High volume of White blood cells is correlated with CKD.
- Presence of 



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
CLI used :
```bash
python -m src.cli --input "C:\Users\devil\Documents\Data_Science\CDK_project\01.RAW_DATA\Chronic_Kidney_Disease\chronic_kidney_disease_full.xlsx" -t "Class" -a "clustering"
```

From the elbow curve displayed, we choose 3 clusters and deduce the following results. We interpret significant variables only, at a threshold of 0.99 for the p-value.
               Variable       P-Value Cluster     Mean/Mode
                Sodium  3.431283e-03       0    130.462093
   WhiteBloodCellCount  9.565631e-29       0   5868.421053
               PusCell  7.562666e-03       0      1.000000
             BloodUrea  2.884129e-03       1     65.393533
                Sodium  1.596533e-03       1    135.220000
            Hemoglobin  1.464878e-03       1     10.905669
      PackedCellVolume  8.683747e-03       1     33.606932
   WhiteBloodCellCount  3.648662e-03       1   9402.194891
     RedBloodCellCount  2.199797e-03       1      4.022463
               PusCell  7.562666e-03       1      1.000000
   WhiteBloodCellCount  3.841947e-37       2  15972.222222
               PusCell  7.562666e-03       2      0.000000

From the table above, and when comparing with the EDA we did in the first notebook, there seem to be 3 sub-types of CKD :

- One related with abnormal PusCells, a low number of White Cells, and a low concetration of Sodium.
- One related with a high concentration of blood urea, and higher concentration of Sodium, Hemoglobin, white and red blood cells.
- One related with a very low number of White Blood cells.

