from numpy import double, set_printoptions
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
import csv


def write_csv(sentences, filename):
    """Writes the the sentences provided, with each word's tags, into a CSV file.

        :param sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence, including start and end tags.
        :param filename: The name of the CSV file to write to.
    """
    with open(filename, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        file_writer.writerow(['Token', 'POS'])
        for sentence in sentences:
            for (word, tag) in sentence:
                if word != "<s>" and word != "</s>" and word != "<UNK>":
                    file_writer.writerow([word, tag])
            file_writer.writerow(['', ''])


def read_csv(filename):
    """Reads a CSV file and iterates through it extracts each sentence's words and tags from the file.

        :param filename: The name of the CSV file to read from.
        
        :return sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence, including start and end tags.
        :return word_count: A dic of word to integer, which is the frequency of each words occurence.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)
    sentences = []
    sen = []
    word_count = defaultdict(int)
    sen.append(("<s>", "START"))
    word_count["</s>"] += 1
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                if row[0] == '':
                    word_count["</s>"] += 1
                    sen.append(("</s>", "END"))
                    sentences.append(sen)

                    sen = []
                    sen.append(("<s>", "START"))
                    word_count["<s>"] += 1
                else:
                    word_count[row[0].lower()] += 1
                    sen.append((row[0].lower(), row[1]))
    word_count["</s>"] += 1
    sen.append(("</s>", "END"))
    sentences.append(sen)
    return sentences, word_count


def split_train_dev(sentences, percentage):
    """Splits the sentences into the training set and the dev set.

        :param sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence.
        :param percentage: The percentage of sentences to go into the training set.
        
        :return train: A 2d array of tuples in the form of (word,tag) for each word of each sentence in the training set.
        :return dev: A 2d array of tuples in the form of (word,tag) for each word of each sentence in the dev set.
    """
    train, dev = train_test_split(sentences, test_size=percentage)

    return train, dev


def convert_to_unk(sentences, word_counts, threshold):
    """Iterates through each word in each sentences and replaces words with a frequency lower then the threshold to the word UNK.

        :param sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence.
        :param word_counts: A dict of word to integer, which is the frequency of each words occurence.
        :param threshold: An int that dicates the minimum frequency a word must have before being replaces with UNK.

        :return new_sentences: The new 2d array of tuples in the form of (word,tag) for each word of each sentence.
    """
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for (word, tag) in sentence:
            if word_counts[word] > threshold:
                new_sentence.append((word, tag))
            else:
                new_sentence.append(("<UNK>", tag))
        new_sentences.append(new_sentence)

    return new_sentences


def get_vocab_counts(training_sentences):
    """Iterates through the words and their tags, stores and returns counts needed for First Order HMM POS.

        :param training_sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence.

        :return total_tag_counts: A dict of each tag's frequency.
        :return double_tag_counts: A dict of each occurence of two consecutive tags and their frequencies.
        :return word_tag_counts: A dict of (word, tag) pairing and their frequency.
    """
    total_tag_counts = defaultdict(int)
    double_tag_counts = defaultdict(int)
    word_tag_counts = defaultdict(int)

    for sentence in training_sentences:
        #Count the start tag, and set it to prev_tag
        total_tag_counts["START"] += 1
        prev_tag = "START"
        word_tag_counts[("<s>", "START")] += 1

        for i in range(1, len(sentence)):

            (word, tag) = sentence[i]
            total_tag_counts[tag] += 1
            double_tag_counts[(prev_tag, tag)] += 1
            word_tag_counts[(word, tag)] += 1
            prev_tag = tag

    return total_tag_counts, double_tag_counts, word_tag_counts


def get_emission_prob(word, tag, word_tag_counts, total_tag_counts):
    """Calcualtes the probability of a given tag occuring given a word.

        :param word: The given word.
        :param tag: The given tag.
        :param word_tag_counts: A dict of (word, tag) pairing and their frequency.
        :param total_tag_counts: A dict of each tag's frequency.

        :return: The probability (float) of the tag occuring given the word.
    """
    if ((word, tag) in word_tag_counts):
        return word_tag_counts[(word, tag)]/total_tag_counts[tag]
    else:
        return 0


def get_emission_prob_table(word_tag_counts, total_tag_counts):
    """Contructs a dict of each word tag pair and their respective emmission probabilities.

        :param word_tag_counts: A dict of (word, tag) pairing and their frequency.
        :param total_tag_coutns: A dict of each tag's frequency.

        :return table: A dict of dicts containing word tag pairs and their respective emission probabilities.
    """
    table = defaultdict(lambda: defaultdict(float))
    for (word, _) in word_tag_counts:

        for tag in total_tag_counts:
            table[word][tag] = get_emission_prob(
                word, tag, word_tag_counts, total_tag_counts)

    return table


def get_bigram_prob_laplace(prev_tag, tag, double_tag_counts, total_tag_counts):
    """Calculates the probability of a tag occuring given another tag (bigram probability), using laplace (add 1) smoothing.

        :param prev_tag: The previous tag in the sentence.
        :param tag: The current tag in the sentence.
        :param double_tag_counts: A dict of each occurence of two consecutive tags and their frequencies.
        :param total_tag_counts: A dict of each tag's frequency.

        :return: The probability (float) of the given tag occuring given the previous tag.
    """
    if ((prev_tag, tag) in double_tag_counts):
        return (double_tag_counts[(prev_tag, tag)]+1)/(total_tag_counts[prev_tag] + len(total_tag_counts))
    else:
        return 1/(total_tag_counts[prev_tag] + len(total_tag_counts))


def get_bigram_prob_table(double_tag_counts, total_tag_counts, bigram_prob_func):
    """Contructs a dict of each tag tag pair and their respective bigram probabilities.

        :param double_tag_counts: A dict of each occurence of two consecutive tags and their frequencies.
        :param total_tag_counts: A dict of each tag's frequency.
        :param bigram_prob_func: The function to be used when calculating bigram probabilites

        :return table: A dict of dicts containing tag tag pairs and their respective bigram probabilities.
    """
    table = defaultdict(lambda: defaultdict(float))
    for prev_tag in total_tag_counts:
        for tag in total_tag_counts:
            table[prev_tag][tag] = bigram_prob_func(
                prev_tag, tag, double_tag_counts, total_tag_counts)

    return table


def get_unigram_prob_table(total_tag_counts):
    """Calculates and contructs a dict of tags and the prbabilites of occuring.

        :param total_tag_counts: A dict of each tag's frequency.

        :return table: A dict of tags and the probabilies occuring.
    """
    total = 0
    for tag in total_tag_counts:
        total += total_tag_counts[tag]

    table = defaultdict(float)
    for tag in total_tag_counts:
        table[tag] = total_tag_counts[tag]/total

    return table


def viterbi(sentence, double_tag_counts, word_tag_counts, total_tag_counts, bi_lambda, uni_lambda):
    """Implementation of the first order Viterbi algorithm, which is used to tag unseen words in a first order HMM.

        :param sentence: An array of words in the sentence to be tagged.
        :param double_tag_counts: A dict of each occurence of two consecutive tags and their frequencies.
        :param word_tag_counts: A dict of (word, tag) pairing and their frequency.
        :param total_tag_counts: A dict of each tag's frequency.
        :param bi_lambda: The weighting (float) given to bigram proabilities.
        :param uni_lambda: The weighting (float) given to unigram probabilities. 

        :return result: An array of tuples in the form of (word,tag).
    """

    #Check to make sure that the lambda's aren't greater than 1.
    lambda_sum = bi_lambda+uni_lambda
    if lambda_sum > 1:
        bi_lambda /= lambda_sum
        uni_lambda /= lambda_sum
        
    pi = [defaultdict(float)] 
    bp = [{}] #back pointer
    emission_table = get_emission_prob_table(word_tag_counts, total_tag_counts)
    unigram_table = get_unigram_prob_table(total_tag_counts)
    bigram_table = get_bigram_prob_table(
        double_tag_counts, total_tag_counts, get_bigram_prob_laplace)

    #Set initial conditions for viterbi algorithm
    prev_tag = "START"
    pi[0][prev_tag] = 1

    for i in range(1, len(sentence)): #Iterate over every word in sentence
        word = sentence[i]
        pi.append(defaultdict(float)) #Add column to pi and bp
        bp.append({})

        for tag in total_tag_counts: #Nested for loop to check every combination of tags

            for prev_tag in total_tag_counts:

                #Calculate probability using interpolation smoothing 
                prob = (
                    emission_table[word][tag] * ((bi_lambda * bigram_table[prev_tag][tag]) + (uni_lambda*unigram_table[tag])) * pi[i-1][prev_tag])

                #If bigger than current max, then update current max
                if (pi[i][tag] < prob):
                    pi[i][tag] = prob
                    bp[i][tag] = prev_tag


    result = []
    tag = "END"
    #Iterate backwards through the back pointers to find most probable path.
    for i in range(len(sentence)-1, 0, -1):
        result.append((sentence[i], tag))
        tag = bp[i][tag]
    result.append(("<s>", "START"))

    #Return reversed list of (word, tag) and so that it starts from the beginning
    return result[::-1]


def eval(result, ground_truth):
    """Compares the HMM POS result with the actual result and returns the number of correct tags and the number of total tags in the sentence.

        :param result: An array of tuples in the form of (word,tag).
        :param ground_truth: An array of tuples in the form of (word,tag).

        :return count_true: The total number (int) of correct tags.
        :return len_ground_truth: The total number (int) of tags in the sentence .
    """
    count_true = 0
    for i, (word, pos) in enumerate(result):
        (true_word, true_pos) = ground_truth[i]
        if word == true_word and pos == true_pos:
            count_true += 1

    len_ground_truth = len(ground_truth)
    return count_true, len_ground_truth


def get_second_order_counts(training_sentences):
    """Iterates through the words and their tags, stores and returns additinal counts needed for Second Order HMM POS.

        :param training_sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence.

        :return triple_tag_counts: A dict of each occurence of three consecutive tags and their frequencies.
    """

    triple_tag_counts = defaultdict(int)

    for i in range(len(training_sentences)):

        prev_prev_tag = "START"

        (_, prev_tag) = training_sentences[i][1]

        for j in range(2, len(training_sentences[i])):
            (_, tag) = training_sentences[i][j]
            triple_tag_counts[(prev_prev_tag, prev_tag, tag)] += 1
            prev_prev_tag = prev_tag
            prev_tag = tag

    return triple_tag_counts


def get_trigram_laplace_prob(prev_prev_tag, prev_tag, tag, triple_tag_counts, bigram_count, total_tag_counts):
    """Calculates the probability of a tag occuring given two previous tags (trigram probability), using laplace (add 1) smoothing.

        :param prev_prev_tag: The tag prior to the previous tag in the sentence.
        :param prev_tag: The previous tag in the sentence.
        :param tag: The current tag in the sentence.
        :param triple_tag_counts: A dict of each occurence of three consecutive tags and their frequencies.
        :param total_tag_counts: A dict of each tag's frequency.

        :return: The probability (float) of the given tag occuring given two previous tags.
    """

    if ((prev_prev_tag, prev_tag, tag) in triple_tag_counts):

        return (triple_tag_counts[(prev_prev_tag, prev_tag, tag)]+1)/(bigram_count[prev_prev_tag][prev_tag] + len(total_tag_counts))

    else:
        return 1/(bigram_count[prev_prev_tag][prev_tag] + len(total_tag_counts))


def get_trigram_prob_table(triple_tag_counts, bigram_table, total_tag_counts, trigram_prob_func):
    """Contructs a dict of each occurence of three consecutive tags and their respective trigram probabilities.

        :param triple_tag_counts: A dict of each occurence of three consecutive tags and their frequencies.
        :param bigram_table: A dict of each occurence of two consecutive tags and their respective bigram probabilities.
        :param total_tag_counts: A dict of each tag's frequency.
        :param trigram_prob_func: The function to be used when calculating trigram probabilites

        :return table: A dict of each tag tag pair and their respective bigram probabilities.
    """
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for prev_prev_tag in total_tag_counts:
        for prev_tag in total_tag_counts:
            for tag in total_tag_counts:
                table[prev_prev_tag][prev_tag][tag] = trigram_prob_func(
                    prev_tag, prev_tag, tag, triple_tag_counts, bigram_table, total_tag_counts)

    return table


def viterbi_trigram(sentence, double_tag_counts, triple_tag_counts, word_tag_counts, total_tag_counts, tri_lambda, bi_lambda, uni_lambda):
    """Implementation of the second order Viterbi algorithm, which is used to tag unseen words in a second order HMM.

        :param sentence: An array of words in the sentence to be tagged.
        :param double_tag_counts: A dict of each occurence of two consecutive tags and their frequencies.
        :param triple_tag_counts: A dict of each occurence of three consecutive tags and their frequencies.
        :param word_tag_counts: A dict of (word, tag) pairing and their frequency.
        :param total_tag_counts: A dict of each tag's frequency.
        :param tri_lambda: The weighting (float) given to trigram proabilities.
        :param bi_lambda: The weighting (float) given to bigram proabilities.
        :param uni_lambda: The weighting (float) given to unigram probabilities. 

        :return result: An array of tuples in the form of (word,tag).
    """
    # Normalize the lambda values to sum to one
    lambda_sum = tri_lambda+bi_lambda+uni_lambda
    if lambda_sum > 1:
        tri_lambda /= lambda_sum
        bi_lambda /= lambda_sum
        uni_lambda /= lambda_sum

    pi = [defaultdict(float)]
    bp = [{}]
    emission_table = get_emission_prob_table(word_tag_counts, total_tag_counts)
    unigram_table = get_unigram_prob_table(total_tag_counts)
    bigram_table = get_bigram_prob_table(
        double_tag_counts, total_tag_counts, get_bigram_prob_laplace)
    trigram_table = get_trigram_prob_table(
        triple_tag_counts, bigram_table, total_tag_counts, get_trigram_laplace_prob)

    pi[0][("START", "START")] = 1

    # SLIGHTLY REDUCE NUMBER OF LOOPS NECESSARY
    def tag_subsets(k):
        if k in (-1, 0):
            return {"START"}
        else:
            return total_tag_counts
    # iterate through the words in the sentence
    for i in range(1, len(sentence)):
        word = sentence[i]
        pi.append(defaultdict(float))
        bp.append({})
        # iterate through all possible previous tags 
        for prev_tag in tag_subsets(i-1):
            #iterate through all possible current tags
            for tag in total_tag_counts:
                # iterate through all possible previous previous tag
                for prev_prev_tag in tag_subsets(i-2):
                    if emission_table[word][tag] != 0:
                        prob = pi[i-1][(prev_prev_tag, prev_tag)] * \
                            ((tri_lambda * trigram_table[prev_prev_tag][prev_tag][tag]) + (bi_lambda * bigram_table[prev_tag][tag]) + (uni_lambda*unigram_table[tag])) * \
                            emission_table[word][tag]

                        if prob > pi[i][(prev_tag, tag)]:
                            pi[i][(prev_tag, tag)] = prob
                            bp[i][(prev_tag, tag)] = prev_prev_tag

    #Calculate the max two last tags in sentence
    max_val = float('-Inf')
    prev_max_tag, tag_max_tag = None, None
    for prev_tag in total_tag_counts:
        for tag in total_tag_counts:
            prob = pi[len(sentence)-1][(prev_tag, tag)] * \
                trigram_table[prev_tag][tag]["END"]

            if prob > max_val:
                max_val = prob
                prev_max_tag = prev_tag
                tag_max_tag = tag

    tag = tag_max_tag
    prev_tag = prev_max_tag

    # Backprop to build tags 
    result = []
    for i in range(len(sentence)-1, 0, -1):
        result.append((sentence[i], tag))
        temp = prev_tag
        prev_tag = bp[i][(prev_tag, tag)]
        tag = temp
    result.append(("<s>", "START"))

    return result[::-1]
