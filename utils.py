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
    sen.append(("<s>","<s>"))   
    for i in range (0, len(pos)):
        if (pd.isna(pos[i])):
            sen.append(("</s>","</s>"))
            sentences.append(sen)

            sen = []   
            sen.append(("<s>","<s>"))   
        else:
            sen.append((tokens[i], pos[i]))
    sen.append(("</s>","</s>"))
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
    

    for sentence in training_sentences:
        total_tag_counts["<s>"]+=1
        prev_tag = "<s>"
        for i,(word,tag) in enumerate(sentence):
            if i == 0:
                continue
            total_tag_counts[tag]+=1
            double_tag_counts[(prev_tag,tag)]+=1
            word_tag_counts[(word,tag)]+=1
            prev_tag = tag

    return total_tag_counts, double_tag_counts, word_tag_counts


def get_bigram_prob_laplace(prev_tag,tag,double_tag_counts,total_tag_counts):
    if ((prev_tag,tag) in double_tag_counts):
        return (double_tag_counts[(prev_tag,tag)]+1)/(total_tag_counts[prev_tag] + len(total_tag_counts))
    else:
        return 0


def get_emission_prob(word,tag,word_tag_counts,total_tag_counts):
    if ((word,tag) in word_tag_counts):
        return word_tag_counts[(word,tag)]/total_tag_counts[tag]
    else:
        return 0

def get_emission_prob_table(word_tag_counts,total_tag_counts):
    table = defaultdict(lambda : defaultdict(float))
    for (word,_) in word_tag_counts:
        
        for tag in total_tag_counts:
            table[word][tag] = get_emission_prob(word,tag,word_tag_counts,total_tag_counts)

        
    return table

def get_bigram_prob_table(double_tag_counts,total_tag_counts,bigram_prob_func):
    table = defaultdict(lambda : defaultdict(float))
    for (prev_tag,_) in double_tag_counts:
        
        for tag in total_tag_counts:
            table[prev_tag][tag] = bigram_prob_func(prev_tag,tag,double_tag_counts,total_tag_counts)
       
    return table

def linear_interpolation_smoothing(bi_lamba,bi_prob,uni_lambda,uni_prob):
    return bi_lamba*bi_prob+uni_lambda*uni_prob

# what about y0 == <s>
def viterbi(sentence, double_tag_counts, word_tags_counts, total_tag_counts):
    pi = defaultdict(lambda: defaultdict(float))
    bp = defaultdict(dict)
    emission_table = get_emission_prob_table(word_tags_counts,total_tag_counts)
    bigram_table = get_bigram_prob_table(double_tag_counts,total_tag_counts,get_bigram_prob_laplace)

    max_tag = ""
    prev_word = "<s>"
    prev_tag = "<s>"
    pi[prev_word][prev_tag] = 1
    
    for word in (sentence):
        
        for tag in total_tag_counts:
            
            for prev_tag in total_tag_counts:
               
                prob = (emission_table[word][tag] * bigram_table[prev_tag][tag] * pi[prev_word][prev_tag])
                
                if (pi[word][tag] < prob):
                    pi[word][tag] = prob
                    max_tag = prev_tag

            bp[word][tag] = max_tag

        prev_word = word

    result = []
    tag = "</s>"
    for i in range(len(sentence)-1,-1,-1):
        print("WORD:",sentence[i])
        print("TAG:",tag)
        result.append((sentence[i],tag))
        tag = bp[sentence[i]][tag]
        print(bp[sentence[i]][tag])
        

    return result[::-1]


def eval(result,ground_truth):
    count_true = 0
    for i,(word,pos) in enumerate(result):
        (true_word,true_pos) = ground_truth[i]
        if word == true_word and pos == true_pos:
            count_true+=1
    
    return count_true-2,len(ground_truth)
