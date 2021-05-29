# POS-HMM
Assignment 1: Part-of-Speech Tagging with Hidden Markov Models
CHNROY002 & TLDEDA001

*******************************CONTENT*******************************

The file handed in contains the following files/folders:
	- main.py
	- utils.py
    - TestSet.csv
    - Training.csv

main.py is the driver which contains the main method and is responsible collecting the commandline parameters and calling the appropriate methods in utils.py

The file utils.py contains all the functions required for the HMM POS tagger.

*********************COMPILATION AND EXECUTION**********************

Compiling and invoking the HMM POS tagger is done by running the following on the command line:

    - python3 main.py <threshold>

Where threshold is a number greater than 0, which decides the frequency threshold in which a word would be consiered unknown and become "<UNK>"


*****************************EXTENSION******************************

One could switch from the defualt first order HMM tagger to the second order HMM tagger by adding the -s flag when invoking the programme as such:

    - python3 main.py <threshold> -s
