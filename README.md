# Naive-Bayes-Classifier-for-spam-filtering
This is a project of the course "Introduction to AI" at Concordia Univeristy
(1)Install: 
Before running the code, the environment should have installed the following library: math, re, numpy, matplotlib.pyplot 
(2) choose experiment:
After running, the program will require:  
Input the No. of experiment need to implement(1,2,3, 4 or 5):
  1: baseline experiment;
  2: stop-words filtering
  3: word lenght filtering
  4: infrequent word filtering: 
  First, gradually remove words with frequency=1, <=5,<=10,<=15 and <=20. get a vocabulary_1;
  Second, sort vocabulary_1 by word frequenct, gradually remove top 5%, 10%, 15%, 20% and 25% most frequent words of vocabulary_1 . 
   (In this experiment, the number of top 5% most frequent words = len(vocabulary_ 1)*5%.)
  5: smoothing with value from 0 to 1 with a step of 0.1

