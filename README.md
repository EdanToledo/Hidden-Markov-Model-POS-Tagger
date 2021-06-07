# POS-HMM
## Assignment 1: Part-of-Speech Tagging with Hidden Markov Models
### CHNROY002 & TLDEDA001

**CONTENTS**

The file handed in contains the following files/folders:
```
- main.py - contains method to run tagger on data.
- utils.py - contains all methods to construct model.
- Split_Train.csv - Training data used for evaluating on dev set.
- Split_Dev.csv - Dev set.
- TestSet.csv - Testing set.
- Training.csv - Training set.
```

**main.py** is the driver which contains the main method and is responsible collecting the commandline parameters and calling the appropriate methods in **utils.py**

The file **utils.py** contains all the functions required for the HMM POS tagger.

We tried to code with modularity in mind so every function can be used on it's own and allow easy and quick adaptability. 

### **COMPILATION AND EXECUTION**

Compiling and invoking the HMM POS tagger is done by running the following on the command line:

```bash
python3 main.py <arguments>
```
e.g
```
python3 main.py --unk_threshold 1 --use_trigram --tri_lambda 0.4 --bi_lambda 0.4 --uni_lambda 0.2
```

#### Arguments
```
usage: main.py [-h] [--unk_threshold UNK_THRESHOLD]
               [--training_file TRAINING_FILE] [--testing_file TESTING_FILE]
               [--use_trigram] [--log_wandb] [--tri_lambda TRI_LAMBDA]
               [--bi_lambda BI_LAMBDA] [--uni_lambda UNI_LAMBDA]
```
```
optional arguments:
  -h, --help            show this help message and exit
  --unk_threshold UNK_THRESHOLD, -u UNK_THRESHOLD
                        Lower bound of word frequency before being regarded as
                        <UNK>
  --training_file TRAINING_FILE, -tr TRAINING_FILE
                        Name of training file
  --testing_file TESTING_FILE, -te TESTING_FILE
                        Name of testing file
  --use_trigram, -ut    Use trigram HMM model
  --log_wandb, -lw      Log to weights and biases platform
  --tri_lambda TRI_LAMBDA, -tl TRI_LAMBDA
                        lambda value for trigram probability in interpolation
                        smoothing
  --bi_lambda BI_LAMBDA, -bl BI_LAMBDA
                        lambda value for bigram probability in interpolation
                        smoothing
  --uni_lambda UNI_LAMBDA, -ul UNI_LAMBDA
                        lambda value for unigram probability in interpolation
                        smoothing
```
#### **EXTENSION**

One could switch from the default first order HMM tagger to the second order HMM tagger by adding the -ut flag when invoking the programme as such:

    - python3 main.py -ut

Both first order and second order HMMs have interpolation smoothing, if you dont want to use it - for first order please set lambda values of unigram to zero and for second order set bigram and unigram lambda values to zero. All lambda values are normalised to sum to one so any value can be input and the ratio between them is used.