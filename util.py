import urllib.request as ur
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants

VERBOSE=True

def load_adult_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    adultDF = pd.read_csv(ur.urlopen(url), sep=',', index_col=False, names=constants.COL_NAMES, dtype=constants.TYPE_DICT_COLS)

    return adultDF


def plot_data(df, title="ML Data", xlabel="blah", ylabel="blahblah", fileName="Data.png", myColor=['b', 'k']):
    plt.figure(figsize=(14, 6))
    plt.title(title, fontsize=14)
    sns.countplot(y=df[df.columns[0]], hue=df[df.columns[1]])
    plt.savefig('./plots/' + fileName)

    #plt.show()