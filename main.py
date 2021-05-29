import utils
import pandas as pd

def main():
    sentences = utils.read_csv("Training.csv")
   
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(sentences)
    
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(sentences):
        sentence_words = [word for (word,_) in sentence]
        result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
        
        count,total = (utils.eval(result,sentence))
        tot_count += count
        tot_ground +=total
        print("The models shows a",tot_count/tot_ground*100, " percentage accuracy!")
        
    
if __name__ == "__main__":
    main()