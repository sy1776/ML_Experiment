import pandas as pd
from util import plot_data

VERBOSE = True
WIDTH=800

pd.set_option('display.width', WIDTH)
#np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

def describe_data(df):
    print("Total Instances: %s, Total Features: %s" % (df.shape[0], df.shape[1]))
    print("")
    print("first 5 data: ", df.head(5))
    print("")
    print(df.describe())
    print("")
    null_cols = df.columns[df.isnull().any()]
    print("null_features = ", null_cols.shape)
    print("")
    print("income unique value: ", df['income'].unique())
    #workclass_income = df.groupby('workclass')
    #print('workclass = ', workclass_income.head(5))
    print("")
    print("Number of unique values in each feature:")
    print(df.nunique())
    print("")
    print("workclass unique values and counts:")
    print(df['workclass'].value_counts())
    print("")
    print("occupation unique values and counts:")
    print(df['occupation'].value_counts())
    print("")
    print("relationship unique values and counts:")
    print(df['relationship'].value_counts())
    print("")
    print("occupation unique value: ", df['occupation'].unique())
    print("")
    print("workclass unique value: ", df['workclass'].unique())
    print("")
    print("native_country unique value: ", df['native_country'].unique())
    explore_data(df)

def explore_data(df1):
    colors = ['b', 'g', 'r']
    workclassDF = df1.loc[:, ['workclass', 'income']]

    #Plot the income according to workclass
    plot_data(workclassDF, "Income as per workclass", "count", "workclass", "income_workclass.png", colors)

    occupationDF = df1.loc[:, ['occupation', 'income']]
    #Plot the income according to occupation
    plot_data(occupationDF, "Income as per occupation", "count", "occupation", "income_occupation.png", colors)

    relationshipDF = df1.loc[:, ['relationship', 'income']]
    #Plot the income according to relationship
    plot_data(relationshipDF, "Income as per relationship", "count", "relationship", "income_relationship.png", colors)

    martialstatusDF = df1.loc[:, ['marital_status', 'income']]
    #Plot the income according to marital_status
    plot_data(martialstatusDF, "Income as per marital status", "count", "relationship", "income_marital_status.png", colors)

    educationDF = df1.loc[:, ['education', 'income']]
    #Plot the income according to education
    plot_data(educationDF, "Income as per education", "count", "education", "income_education.png", colors)