# POS-HMM
## Assignment 1: Part-of-Speech Tagging with Hidden Markov Models
### CHNROY002 & TLDEDA001

**CONTENTS**
The file handed in contains the following files/folders:
```
- main.py
- utils.py
- TestSet.csv
- Training.csv
```

**main.py** is the driver which contains the main method and is responsible collecting the commandline parameters and calling the appropriate methods in **utils.py**

The file **utils.py** contains all the functions required for the HMM POS tagger.

We tried to code with modularity in mind so every function can be used on it's own and allow easy and quick adaptability. 

### **COMPILATION AND EXECUTION**

Compiling and invoking the HMM POS tagger is done by running the following on the command line:

```bash
python3 main.py <arguments>
```


#### Arguments
```bash
usage: main.py [-h] [--unk_threshold UNK_THRESHOLD]
               [--training_file TRAINING_FILE]
               [--testing_file TESTING_FILE]
               [--use_trigram USE_TRIGRAM]
```
```
optional arguments:
  -h, --help 
  show this help message and exit
  
  --unk_threshold UNK_THRESHOLD, -u UNK_THRESHOLD
  Lower bound of word frequency before being regarded as <UNK>
  
  --training_file TRAINING_FILE, -tr TRAINING_FILE
  Name of training file
  
  --testing_file TESTING_FILE, -te TESTING_FILE
  Name of testing file
  
  --use_trigram USE_TRIGRAM, -ut
  Use trigram HMM model
```

#### **EXTENSION**

One could switch from the defualt first order HMM tagger to the second order HMM tagger by adding the -ut flag when invoking the programme as such:

    - python3 main.py -ut
