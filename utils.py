from numpy import double, set_printoptions
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from collections import defaultdict
import math
import csv

def read_csv(filename):
    sentences = []
    sen = []
    word_count = defaultdict(int)
    sen.append(("<s>","START")) 
    word_count["</s>"]+=1
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter =",")

        for i,row in enumerate(csv_reader):
            if i == 0:
                pass
            else:
                if row[0] == '':
                    word_count["</s>"]+=1
                    sen.append(("</s>","END"))
                    sentences.append(sen)

                    sen = []   
                    sen.append(("<s>","START")) 
                    word_count["<s>"]+=1
                else:
                    word_count[row[0].lower()]+=1
                    sen.append((row[0].lower(), row[1]))
    word_count["</s>"]+=1
    sen.append(("</s>","END"))
    sentences.append(sen) 
    return sentences , word_count


def split_train_dev(sentences, percentage):
    train, dev = train_test_split(sentences, test_size=percentage)

    return train, dev

def convert_to_unk(sentences, word_counts,threshold):
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for (word,tag) in sentence:
            if word_counts[word]>threshold:
                new_sentence.append((word,tag))
            else:
                new_sentence.append(("<UNK>",tag))
        new_sentences.append(new_sentence)
    
    return new_sentences 



def get_vocab_counts(training_sentences):
    total_tag_counts = defaultdict(int)
    double_tag_counts = defaultdict(int)
    word_tag_counts = defaultdict(int)

    for sentence in training_sentences:
        total_tag_counts["START"]+=1
        prev_tag = "START"
        word_tag_counts[("<s>","START")]+=1
        for i in range(1,len(sentence)):
            (word,tag) = sentence[i]
            total_tag_counts[tag]+=1
            double_tag_counts[(prev_tag,tag)]+=1
            word_tag_counts[(word,tag)]+=1
            prev_tag = tag

    return total_tag_counts, double_tag_counts, word_tag_counts


def get_bigram_prob_laplace(prev_tag,tag,double_tag_counts,total_tag_counts):
    if ((prev_tag,tag) in double_tag_counts):
        return (double_tag_counts[(prev_tag,tag)]+1)/(total_tag_counts[prev_tag] + len(total_tag_counts))
    else:
        return 1/(total_tag_counts[prev_tag] + len(total_tag_counts))


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
    for prev_tag in total_tag_counts:
        for tag in total_tag_counts:
            table[prev_tag][tag] = bigram_prob_func(prev_tag,tag,double_tag_counts,total_tag_counts)
       
    return table

def linear_interpolation_smoothing(bi_lamba,bi_prob,uni_lambda,uni_prob):
    return bi_lamba*bi_prob+uni_lambda*uni_prob


def viterbi(sentence, double_tag_counts, word_tag_counts, total_tag_counts):
    pi = [defaultdict(float)]
    bp = [{}]
    emission_table = get_emission_prob_table(word_tag_counts,total_tag_counts)
    bigram_table = get_bigram_prob_table(double_tag_counts,total_tag_counts,get_bigram_prob_laplace)

    prev_tag = "START"
    pi[0][prev_tag] = 1
    
    for i in range(1,len(sentence)):
        word = sentence[i]
        pi.append(defaultdict(float))
        bp.append({})
        for tag in total_tag_counts:
            
            for prev_tag in total_tag_counts:
               
                prob = (emission_table[word][tag] * bigram_table[prev_tag][tag] * pi[i-1][prev_tag])
               
                # if pi[i-1][prev_tag]>0:
                #     print("WORD NUMBER:",i,"out of",len(sentence))
                #     print(word,tag,"emmission:",emission_table[word][tag])
                #     print(prev_tag,tag,"bigram:",bigram_table[prev_tag][tag])
                #     print(sentence[i-1],prev_tag,"pi:",pi[i-1][prev_tag])

                if (pi[i][tag] < prob):
                    pi[i][tag] = prob
                    bp[i][tag] = prev_tag

            
    result = []
    tag = "END"
    for i in range(len(sentence)-1,0,-1):
        result.append((sentence[i],tag))
        tag = bp[i][tag]
    result.append(("<s>","START"))

    return result[::-1]

def eval(result,ground_truth):
    count_true = 0
    for i,(word,pos) in enumerate(result):
        (true_word,true_pos) = ground_truth[i]
        # if pos!=true_pos:
        #     print(pos,true_pos)
        if word == true_word and pos == true_pos:
            count_true+=1
    
    return count_true,len(ground_truth)

def get_second_order_counts(training_sentences):
    total_tag_counts = defaultdict(int)
    triple_tag_counts = defaultdict(int)
    word_tag_counts = defaultdict(int)

    for i in range(len(training_sentences)):
        total_tag_counts["START"]+=1
        prev_prev_tag = "START"
        word_tag_counts[("<s>","START")]+=1

        (word,prev_tag) = training_sentences[i][1]
        total_tag_counts[prev_tag] += 1
        word_tag_counts[(word,prev_tag)]+= 1

        for j in range(2,len(training_sentences[i])):
            (word,tag) = training_sentences[i][j]
            total_tag_counts[tag]+=1
            triple_tag_counts[(prev_prev_tag,prev_tag,tag)]+=1
            word_tag_counts[(word,tag)]+=1
            prev_prev_tag = prev_tag
            prev_tag = tag

    return total_tag_counts, triple_tag_counts, word_tag_counts

def get_trigram_laplace_prob(prev_prev_tag,prev_tag,tag,triple_tag_counts,bigram_count,total_tag_counts):

    if ((prev_prev_tag,prev_tag,tag) in triple_tag_counts):

        return (triple_tag_counts[(prev_prev_tag, prev_tag,tag)]+1)/(bigram_count[prev_prev_tag][prev_tag] + len(total_tag_counts))

    else:

        return 1/(total_tag_counts[prev_tag] + len(total_tag_counts))


def get_trigram_prob_table(triple_tag_counts,bigram_counts,total_tag_counts,trigram_prob_func):
    table = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    for prev_prev_tag in total_tag_counts:
        for prev_tag in total_tag_counts:
            for tag in total_tag_counts:
                table[prev_prev_tag][prev_tag][tag] = trigram_prob_func(prev_tag,prev_tag,tag,triple_tag_counts,bigram_counts,total_tag_counts)
       
    return table

def viterbi_trigram(sentence, double_tag_counts, triple_tag_counts, word_tag_counts, total_tag_counts):
    pi = defaultdict(float)
    bp = defaultdict(float)
    emission_table = get_emission_prob_table(word_tag_counts,total_tag_counts)
    bigram_table = get_bigram_prob_table(double_tag_counts,total_tag_counts,get_bigram_prob_laplace)
    trigram_table = get_trigram_prob_table(triple_tag_counts, bigram_table, total_tag_counts, get_trigram_laplace_prob)

    pi[(0,"START", "START")] = 1
    
    for i in range(1,len(sentence)):
        word = sentence[i]

        for tag in total_tag_counts:
            for prev_tag in total_tag_counts:
                max_score = -1
                max_tag = None
                for prev_prev_tag in total_tag_counts:
                    print("here")
                    prob = emission_table[word][tag] * trigram_table[prev_prev_tag][prev_tag][tag] * pi[(i-1,prev_prev_tag,prev_tag)]
                    print("here")

                    if (max_score < prob):
                        max_score = prob
                        max_tag = prev_prev_tag
                        
                pi[(i,prev_tag,tag)] = max_score
                bp[(i,prev_tag,tag)] = max_tag
            
    result = []
    tag = "END"
    for i in range(len(sentence)-1,0,-1):
        result.append((sentence[i],tag))
        tag = bp[i][tag]
    result.append(("<s>","START"))

    return result[::-1]