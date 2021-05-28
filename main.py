import utils



def main():
    df = utils.read_csv("Training.csv")
    sentences = utils.split_sentences(df)
    
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(sentences)
    
    sentence = sentences[0]
    sentence_words = []
    for (word,tag) in sentence:
        sentence_words.append(word)


    result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
    print(result)
    
if __name__ == "__main__":
    main()