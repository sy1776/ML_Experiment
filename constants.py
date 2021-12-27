COL_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
#Numeric column types need to be 64 bits, 'Int64'. Otherwise, XGBoost will not work. It will throw the error
TYPE_DICT_COLS = {'age': 'int64', 'fnlwgt': 'int64', 'education_num': 'int64', 'capital_gain': 'int64',
                  'capital_loss': 'int64', 'hours_per_week': 'int64'}
STRING_COLS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
