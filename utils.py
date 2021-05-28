from numpy import double
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from collections import defaultdict
import math

def split_sentences(df):
    pos = df['POS']
    tokens = df['Token']
    sentences = []
    sen = []
    for i in range (0, len(pos)):
        if (pd.isna(pos[i])):
            sentences.append(sen)
            print(sen, "\n")
            sen = []      
        else:
            sen.append((tokens[i], pos[i]))
    sentences.append(sen)        

    return sentences

def read_csv(filename):
    dirname = os.path.dirname(__file__)
    relative_file = os.path.join(dirname, filename)
    df = pd.read_csv(relative_file)
    return df


def split_train_dev(sentences, percentage):
    train, dev = train_test_split(sentences, test_size=percentage)

    return train, dev


def get_vocab_counts(training_sentences):
    total_tag_counts = defaultdict(int)
    double_tag_counts = defaultdict(int)
    word_tag_counts = defaultdict(int)
    prev_tag = "<s>"

    for sentence in training_sentences:
        total_tag_counts["<s>"]+=1
        for (word,tag) in sentence:
            total_tag_counts[tag]+=1
            double_tag_counts[(prev_tag,tag)]+=1
            word_tag_counts[(word,tag)]+=1
            prev_tag = tag

        double_tag_counts[(prev_tag,"</s>")] +=1
        total_tag_counts["</s>"]+=1


    return total_tag_counts, double_tag_counts


def get_bigram_prob_laplace(prev_tag,tag,double_tag_counts,total_tag_counts):

    return (double_tag_counts[(prev_tag,tag)]+1)/(total_tag_counts[prev_tag] + len(total_tag_counts))


def get_emission_prob(word,tag,word_tag_counts,total_tag_counts):
    return word_tag_counts[(word,tag)]/total_tag_counts[tag]

def get_emission_prob_table(word_tag_counts,total_tag_counts):
    table = {}
    for (word,_) in word_tag_counts.items():
        table[word] = {}
        for tag in total_tag_counts:
            table[word][tag] = get_emission_prob(word,tag,word_tag_counts,total_tag_counts)

    return table

def get_bigram_prob_table(double_tag_counts,total_tag_counts,bigram_prob_func):
    table = {}
    for (prev_tag,_) in double_tag_counts:
        table[prev_tag] = {}
        for tag in total_tag_counts:
            table[prev_tag][tag] = bigram_prob_func(prev_tag,tag,double,total_tag_counts)

    return table

def get_sentence_tag_prob(bi_gram_prob_func,sentence_tags,double_tag_counts,total_tag_counts):
    prob = 0
    prev_tag = "<s>"
    for tag in sentence_tags:
        prob += math.log(bi_gram_prob_func(prev_tag,tag,double_tag_counts,total_tag_counts))
        prev_tag = tag
    prob += math.log(bi_gram_prob_func(prev_tag,"</s>",double_tag_counts,total_tag_counts))

    return prob
         

def linear_interpolation_smoothing(bi_lamba,bi_prob,uni_lambda,uni_prob):
    return bi_lamba*bi_prob+uni_lambda*uni_prob

# what about y0 == <s>
def viterbi(sentence, double_tag_counts, word_tags_counts, total_tag_counts):
    pi = [[0]*len(total_tag_counts)]*(len(sentence)+1)
    prev_tag = "<s>"
    pi[0][0] = 1
    for i, (word, tag) in enumerate(sentence):
        for j,t in enumerate(total_tag_counts):
            pi[i+1][j] = get_emission_prob(word, t, word_tags_counts, total_tag_counts) * get_bigram_prob_laplace(prev_tag, t, double_tag_counts,total_tag_counts) * pi[i][j]
        prev_tag = tag

    
