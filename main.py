import utils



def main():
    df = utils.read_csv("Training.csv")
    sentences = utils.split_sentences(df)
    
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(sentences)
    
    sentence = sentences[0]
    for (word,tag) in sentence:
        print(word,end=" ")

    print("\n\n\n")
    utils.viterbi(sentence,double_tag_counts,word_tag_counts,total_tag_counts)
    
if __name__ == "__main__":
    main()