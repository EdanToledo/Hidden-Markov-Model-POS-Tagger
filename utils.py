from numpy import double
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from collections import defaultdict


def read_csv(filename):
    dirname = os.path.dirname(__file__)
    relative_file = os.path.join(dirname, filename)
    df = pd.read_csv(relative_file)
    return df


def split_train_dev(df, percentage):
    train, dev = train_test_split(df, test_size=percentage)

    return train, dev


def get_vocab_counts(training_sentences):
    total_tag_counts = defaultdict(int)
    double_tag_counts = defaultdict(int)
    prev_tag = "<s>"

    for sentence in training_sentences:
        for tag in sentence:
            total_tag_counts[tag]+=1
            double_tag_counts[(prev_tag,tag)]+=1
            prev_tag = tag
        double_tag_counts[(prev_tag,"</s>")] +=1
        total_tag_counts["</s>"]+=1


    return total_tag_counts, double_tag_counts
