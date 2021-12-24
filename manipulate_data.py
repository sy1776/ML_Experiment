from sklearn.preprocessing import LabelEncoder
import pandas as pd
import constants

VERBOSE = True

def change_y_to_binary_value(df):
    if VERBOSE:
        print("Replacing category values with binary values: 0 or 1")
        print("Before replacing = ")
        print(df['income'].value_counts())
    df['income'].replace(['<=50K','>50K'], [0,1], inplace=True)

    if VERBOSE:
        print("After replacing = ")
        print(df['income'].value_counts())
    return df

def filter_missing_data(df):
    #There are lots of instances missing with values. They're encoded with '?' in workclass, native country, and occupation.
    # Removing any examples with '?'
    if VERBOSE:
        print("Before filtering, total instances and features: ", df.shape)

    df1 = df.loc[ (df['occupation'] != '?') & (df['workclass'] != '?') & (df['native_country'] != '?') ]
    #print(df['occupation'] != '?')
    if VERBOSE:
        #print(df[ (df['occupation'] == '?') | (df['workclass'] == '?') | (df['native_country'] == '?') ])
        print("After filtering, total instances and features: ", df1.shape)

    return df1

def remove_space(df):
    #Data constains a leading space in every values in all columns. Below will remove a white space
    #on the columns with string (object) type
    for i in constants.STRING_COLS:
        df[i] = df[i].str.strip()
    return df

def find_remove_dups(df):
    dups = df[df.duplicated()]
    print("dups = ", dups.shape[0])
    if (dups.shape[0] > 0):
        dups_with_base = df[df.duplicated(keep=False)]
        dups_with_base.to_csv('dups.csv', sep=',', index=False)
        df = df.drop_duplicates()
        print("After removing dups, total instances and features: ", df.shape)

    return df

def transform_data(df):
    #Feature engineering
    #Transform features with strings to numeric values (0 to n_classes -1)
    le = LabelEncoder()
    df[constants.STRING_COLS] = df[constants.STRING_COLS].apply(le.fit_transform)

    print("transformeddf = ")
    print(df.head(20))
    return df