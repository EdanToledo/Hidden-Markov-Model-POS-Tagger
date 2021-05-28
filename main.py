import utils



def main():
    df = utils.read_csv("Training.csv")
    sentences = utils.split_sentences(df)
    # print(sentences)
    total_tag_counts, double_tag_counts = utils.get_vocab_counts(sentences)
    #print(double_tag_counts)
    
if __name__ == "__main__":
    main()