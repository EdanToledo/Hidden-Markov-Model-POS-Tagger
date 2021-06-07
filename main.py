import utils
import wandb
import argparse


def run(training_file, testing_file, use_trigram, unk_threshold, log_to_wandb, tri_lambda, bi_lambda, uni_lambda):
    if log_to_wandb:
        wandb.init(project="POS-TAGGER-HMM")

    training_sentences, training_word_count = utils.read_csv(training_file)

    # training_sentences, dev_sentences = utils.split_train_dev(training_sentences,0.15)
    # utils.write_csv(training_sentences,"Split_Train")
    # utils.write_csv(dev_sentences,"Split_Dev")

    training_sentences = utils.convert_to_unk(
        training_sentences, training_word_count, unk_threshold)

    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(
        training_sentences)

    if use_trigram:
        triple_tag_counts = utils.get_second_order_counts(training_sentences)

    testing_sentences, _ = utils.read_csv(testing_file)

    testing_sentences = utils.convert_to_unk(
        testing_sentences, training_word_count, unk_threshold)

    tot_count = 0
    tot_ground = 0
    print("Testing on the testing set...\n")
    for i, (sentence) in enumerate(testing_sentences):
        sentence_words = [word for (word, _) in sentence]

        if use_trigram:
            result = utils.viterbi_trigram(sentence_words, double_tag_counts, triple_tag_counts,
                                           word_tag_counts, total_tag_counts, tri_lambda, bi_lambda, uni_lambda)
        else:
            result = utils.viterbi(sentence_words, double_tag_counts,
                                   word_tag_counts, total_tag_counts, bi_lambda, uni_lambda)

        count, total = (utils.eval(result, sentence))
        tot_count += count
        tot_ground += total
        print("The models shows a", tot_count /
              tot_ground*100, " percentage accuracy!")
        if log_to_wandb:
            wandb.log({"accuracy": tot_count/tot_ground*100})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train an HMM-POS tagger and evaluate it on a testing set')
    parser.add_argument('--unk_threshold', "-u", default=1, type=int,
                        help='Lower bound of word frequency before being regarded as <UNK>')
    parser.add_argument('--training_file', "-tr", default="Training.csv",
                        type=str, help='Name of training file')
    parser.add_argument('--testing_file', "-te", default="TestSet.csv",
                        type=str, help='Name of testing file')
    parser.add_argument('--use_trigram', "-ut",
                        action='store_true', help='Use trigram HMM model')
    parser.add_argument('--log_wandb', "-lw", action='store_true',
                        help='Log to weights and biases platform')
    parser.add_argument('--tri_lambda', "-tl", default=1, type=float,
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

    log_to_wandb = True
    use_trigram = True


    run(training_file, testing_file, use_trigram, unk_threshold,
        log_to_wandb, tri_lambda, bi_lambda, uni_lambda)
