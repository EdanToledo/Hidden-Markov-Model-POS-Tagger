from sklearn.model_selection import train_test_split
import pandas as pd
import os


def read_csv(filename):
    dirname = os.path.dirname(__file__)
    relative_file = os.path.join(dirname, filename)
    df = pd.read_csv(relative_file)
    return df

def split_train_dev(df,percentage):
    train, dev = train_test_split(df, test_size=percentage)
    
    return train,dev

