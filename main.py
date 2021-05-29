import utils
import wandb
import argparse



def run(training_file,testing_file,unk_threshold,log_to_wandb):
    if log_to_wandb:
        wandb.init(project="POS-TAGGER-HMM")

    training_sentences ,training_word_count = utils.read_csv(training_file)
    
    training_sentences = utils.convert_to_unk(training_sentences,training_word_count,unk_threshold)

    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(training_sentences)
    
    
    testing_sentences ,_ = utils.read_csv(testing_file)

    testing_sentences = utils.convert_to_unk(testing_sentences,training_word_count,unk_threshold)
   
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(testing_sentences):
        sentence_words = [word for (word,_) in sentence]
        
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
    
    args = parser.parse_args()
    
    unk_threshold = args.unk_threshold
    training_file = args.training_file
    testing_file = args.testing_file

    run(training_file,testing_file,unk_threshold,True)
    