import utils



def main():
    df = utils.read_csv("Training.csv")
    sentences = utils.split_sentences(df)
    total_tag_counts, double_tag_counts, word_tag_counts = utils.get_vocab_counts(sentences)
    
    #df = utils.read_csv("TestSet.csv")
    sentences = utils.split_sentences(df)
    tot_count=0
    tot_ground=0
    print("Testing on the testing set...\n")
    for i,(sentence) in enumerate(sentences):
        # print(i)
        sentence_words = [word for (word,_) in sentence]
        # print(sentence_words)
        result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
        
        count,total = (utils.eval(result,sentence))
        tot_count += count
        tot_ground +=total
        print("The models shows a",tot_count/tot_ground*100, " percentage accuracy!")
        if i == 0:
            break
        
        

    # dftest = utils.read_csv("TestSet.csv")
    # sentences = utils.split_sentences(dftest)

    # tot_count=0
    # tot_ground=0
    # sentence_words = []
    # for sentence in sentences:
    #     for (word,_) in sentence:
    #         sentence_words.append(word)

    #     result = utils.viterbi(sentence_words,double_tag_counts,word_tag_counts,total_tag_counts)
    #     count,total = (utils.eval(result,sentence))
    #     tot_count+=count
    #     tot_ground+=total
    #     sentence_words = []

    #


    
    
if __name__ == "__main__":
    main()