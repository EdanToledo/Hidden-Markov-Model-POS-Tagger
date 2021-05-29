import utils
import pandas as pd

def trigram():
    training_sentences ,training_word_count = utils.read_csv("Training.csv")
   
    training_sentences = utils.convert_to_unk(training_sentences,training_word_count,2)

    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(training_sentences)
    total_tag_counts, triple_tag_counts, word_tag_counts = utils.get_second_order_counts(training_sentences)

    testing_sentences = training_sentences
   
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(testing_sentences):
        print(i)
        sentence_words = [word for (word,_) in sentence]
        
        result = utils.viterbi_trigram(sentence_words, double_tag_counts, triple_tag_counts, word_tag_counts, total_tag_counts)
        
        count,total = (utils.eval(result,sentence))
        tot_count += count
        tot_ground +=total
        print("The models shows a",tot_count/tot_ground*100, " percentage accuracy!")

def main():

    training_sentences ,training_word_count = utils.read_csv("Training.csv")
    #training_sentences,dev_sentences = utils.split_train_dev(training_sentences,0.2)

  
   
    training_sentences = utils.convert_to_unk(training_sentences,training_word_count,2)

    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(training_sentences)
    
    testing_sentences = training_sentences
    
    # testing_sentences ,_ = utils.read_csv("TestSet.csv")

    #testing_sentences = utils.convert_to_unk(testing_sentences,training_word_count,2)
   
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(testing_sentences):
        print(i)
        sentence_words = [word for (word,_) in sentence]
        
        result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
        
        count,total = (utils.eval(result,sentence))
        tot_count += count
        tot_ground +=total
        print("The models shows a",tot_count/tot_ground*100, " percentage accuracy!")
        
        
    
if __name__ == "__main__":
    trigram()