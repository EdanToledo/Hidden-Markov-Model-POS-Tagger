import utils
import wandb
import argparse


def run(training_file, testing_file, use_trigram, unk_threshold, log_to_wandb, tri_lambda, bi_lambda, uni_lambda):
    """Main method to run the HMM POS tagger.

        :param training_file: Name of the CSV file used to train the HMM POS.
        :param testing_file: Name of the CSV file used to test the HMM POS.
        :param use_trigram: Boolean to dictate whether to run a first order (false) or second order (true) HMM POS.
        :param unk_threshold: An int that dicates the minimum frequency a word must have before being replaces with UNK.
        :param log_to_wandb: Boolean to dictate whether or not to log the accuracy to the wandb platform.
        :param tri_lambda: The weighting (float) given to trigram proabilities.
        :param bi_lambda: The weighting (float) given to bigram proabilities.
        :param uni_lambda: The weighting (float) given to unigram probabilities. 
    """
    
    if log_to_wandb:
        wandb.init(project="POS-TAGGER-HMM")

    # get training sentences and the frequency of words from the data
    training_sentences, training_word_count = utils.read_csv(training_file)

    # This is for splitting the dev and training set
    # training_sentences, dev_sentences = utils.split_train_dev(training_sentences,0.15)
    # utils.write_csv(training_sentences,"Split_Train")
    # utils.write_csv(dev_sentences,"Split_Dev")

    # Convert low frequency words into <UNK> words
    training_sentences = utils.convert_to_unk(
        training_sentences, training_word_count, unk_threshold)

    # Get the bigram, unigram and word given tag counts
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(
        training_sentences)

    # if necessary get the trigram counts
    if use_trigram:
        triple_tag_counts = utils.get_second_order_counts(training_sentences)

    # get the testing sentences
    testing_sentences, _ = utils.read_csv(testing_file)

    # convert unseen words to <UNK>
    testing_sentences = utils.convert_to_unk(
        testing_sentences, training_word_count, unk_threshold)
    
    print("Testing on the testing set...\n")
    
    tot_count = 0
    tot_ground = 0
    results = []
    number_of_sentences = len(testing_sentences)
    

    for i, (sentence) in enumerate(testing_sentences):
        sentence_words = [word for (word, _) in sentence]

        if use_trigram:
            result = utils.viterbi_trigram(sentence_words, double_tag_counts, triple_tag_counts,
                                           word_tag_counts, total_tag_counts, tri_lambda, bi_lambda, uni_lambda)
        else:
            result = utils.viterbi(sentence_words, double_tag_counts,
                                   word_tag_counts, total_tag_counts, bi_lambda, uni_lambda)

        results.append(result)
        count, total = (utils.eval(result, sentence))
        tot_count += count
        tot_ground += total
        
        if log_to_wandb:
            wandb.log({"accuracy": tot_count/tot_ground*100})
        print((i/number_of_sentences)*100,"% complete")
    
    # write predicting results
    utils.write_csv(results,"Testing_Prediction.csv")
    print("The models shows a", tot_count / tot_ground*100, " percentage accuracy!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train an HMM-POS tagger and evaluate it on a testing set')
    parser.add_argument('--unk_threshold', "-u", default=1, type=int,
                        help='Lower bound of word frequency before being regarded as <UNK>')
    parser.add_argument('--training_file', "-tr", default="Split_Train.csv",
                        type=str, help='Name of training file')
    parser.add_argument('--testing_file', "-te", default="Split_Dev.csv",
                        type=str, help='Name of testing file')
    parser.add_argument('--use_trigram', "-ut",
                        action='store_true', help='Use trigram HMM model')
    parser.add_argument('--log_wandb', "-lw", action='store_true',
                        help='Log to weights and biases platform')
    parser.add_argument('--tri_lambda', "-tl", default=9, type=float,
                        help='lambda value for trigram probability in interpolation smoothing')
    parser.add_argument('--bi_lambda', "-bl", default=10, type=float,
                        help='lambda value for bigram probability in interpolation smoothing')
    parser.add_argument('--uni_lambda', "-ul", default=1, type=float,
                        help='lambda value for unigram probability in interpolation smoothing')

    args = parser.parse_args()

    unk_threshold = args.unk_threshold
    training_file = args.training_file
    testing_file = args.testing_file
    use_trigram = args.use_trigram
    log_to_wandb = args.log_wandb
    tri_lambda = args.tri_lambda
    bi_lambda = args.bi_lambda
    uni_lambda = args.uni_lambda


    run(training_file, testing_file, use_trigram, unk_threshold,
        log_to_wandb, tri_lambda, bi_lambda, uni_lambda)
