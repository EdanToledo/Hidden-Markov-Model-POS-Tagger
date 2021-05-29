import utils
import wandb
import argparse

def run(training_file,testing_file,use_trigram,unk_threshold,log_to_wandb):
    if log_to_wandb:
        wandb.init(project="POS-TAGGER-HMM")

    training_sentences ,training_word_count = utils.read_csv(training_file)
    
    training_sentences = utils.convert_to_unk(training_sentences,training_word_count,unk_threshold)
   
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(training_sentences)
    
    if use_trigram:
        total_tag_counts, triple_tag_counts, word_tag_counts = utils.get_second_order_counts(training_sentences)

    
    testing_sentences ,_ = utils.read_csv(testing_file)

    testing_sentences = utils.convert_to_unk(testing_sentences,training_word_count,unk_threshold)
   
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(testing_sentences):
        sentence_words = [word for (word,_) in sentence]
        
        if use_trigram:
            result = utils.viterbi_trigram(sentence_words, double_tag_counts, triple_tag_counts, word_tag_counts, total_tag_counts)
        else:
            result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
        
        count,total = (utils.eval(result,sentence))
        tot_count += count
        tot_ground +=total
        print("The models shows a",tot_count/tot_ground*100, " percentage accuracy!")
        if log_to_wandb:
            wandb.log({"accuracy":tot_count/tot_ground*100})
        
        
    
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='Train an HMM-POS tagger and evaluate it on a testing set')
    parser.add_argument('--unk_threshold',"-u",default=1,type=int,help='Lower bound of word frequency before being regarded as <UNK>')
    parser.add_argument('--training_file',"-tr",default="Training.csv",type=str,help='Name of training file')
    parser.add_argument('--testing_file',"-te",default="TestSet.csv",type=str,help='Name of testing file')
    parser.add_argument('--use_trigram',"-ut",default=False,type=bool,help='Use trigram HMM model')
    
    args = parser.parse_args()
    
    unk_threshold = args.unk_threshold
    training_file = args.training_file
    testing_file = args.testing_file
    use_trigram = args.use_trigram

    run(training_file,testing_file,use_trigram,unk_threshold,True)

  
    
