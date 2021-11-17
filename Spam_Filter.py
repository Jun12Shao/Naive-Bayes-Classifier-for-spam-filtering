# -*- coding:utf-8 -*-

"""
@author: Jun
@file: .py
@time: 3/22/20192:01 PM
"""

import math
import re
import numpy as np
import matplotlib.pyplot as plt


def filter_function(str):
    if str=='' or str=='\n':
        return False
    else:
        return True

def get_Vocabulary(filenames):
    vocabulary= {}
    for filename in filenames:
        with open(train_path +filename, 'r', encoding='latin-1') as f1:
            for line in f1.readlines():
                line=line.lower()
                line=list(filter(filter_function,re.split('[^a-zA-Z]',line)))
                for word in line:
                    if word in vocabulary:
                        vocabulary[word]+=1
                    else:
                        vocabulary[word] = 1
        f1.close()

    return vocabulary

def Text_Preprocessing(filename):
    text=[]
    with open(filename, 'r', encoding='latin-1') as f1:
        for line in f1.readlines():
            line = line.lower()
            line = list(filter(filter_function, re.split('[^a-zA-Z]', line)))
            text+=line
    f1.close()

    return text

# Building and evaluating a Naive Bayes Classifier
def Naive_Bayes_Classifier(text,vocabulary,voc_maxtix):
    score1=np.log10(ham_train/(ham_train+spam_train))
    score2=np.log10(spam_train/(ham_train+spam_train))
    for word in text:
        if word in vocabulary:
            index=vocabulary.index(word)
            score1+=np.log10(voc_maxtix[index][1])
            score2+=np.log10(voc_maxtix[index][3])
    if score1>score2:
        c='ham'
    else:
        c='spam'
    return c,score1,score2

def get_PMatrix(vocabulary_list,vocabulary_ham,vocabulary_spam,sm=0.5):
    # get probability Matrix of ham and spam:
    voc_num = len(vocabulary_list)
    voc_maxtix = np.zeros((voc_num, 4), dtype=float)
    for i in range(voc_num):
        word = vocabulary_list[i]
        if word in vocabulary_ham:
            voc_maxtix[i][0] = vocabulary_ham[word]
        else:
            voc_maxtix[i][0] = 0
        if word in vocabulary_spam:
            voc_maxtix[i][2] = vocabulary_spam[word]
        else:
            voc_maxtix[i][2] = 0
    wn_ham = np.sum(voc_maxtix,axis=0)[0]
    print("wn_ham:",wn_ham)
    wn_spam = np.sum(voc_maxtix,axis=0)[2]
    print("wn_spam:", wn_spam)
    for i in range(voc_num):
        voc_maxtix[i][1] = (voc_maxtix[i][0] + sm) / (wn_ham + sm* voc_num)
        voc_maxtix[i][3] = (voc_maxtix[i][2] + sm) / (wn_spam + sm* voc_num)
    return voc_maxtix

def Model_test(test_sets,vocabulary_list, voc_maxtix,result_name='_.txt',w=0):
    i,r,ham = 0,0,1
    T_ham,F_ham,T_spam,F_spam=0,0,0,0
    if w:
        file=open('D:/Projects/AI/project2/' + result_name, 'w', encoding='latin-1')

    for filenames in test_sets:
        if ham:
            label = 'ham'

        else:
            label = 'spam'
        for filename in filenames:
            i += 1
            print(i)
            text = Text_Preprocessing(test_path + filename)
            c, score1, score2 = Naive_Bayes_Classifier(text, vocabulary_list, voc_maxtix)
            if c == label:
                result = 'right'
                r += 1
                if ham:
                    T_ham+=1
                else:
                    T_spam+=1
            else:
                result = 'wrong'
                if ham:
                    F_spam+=1
                else:
                    F_ham+=1
            if w:
                file.write("{}  {}  {}  {}  {}  {}  {}\n".format(i, filename, c, score1, score2, label, result))
        ham = 0
    if w:
        file.close()
    accuracy = r / (ham_test + spam_test)
    print("Accuracy:{},T_ham:{},F_ham:{},T_spam:{},F_spam:{}".format(accuracy, T_ham, F_ham, T_spam, F_spam))

    p_ham = T_ham / (T_ham + F_ham)
    p_spam = T_spam / (T_spam + F_spam)
    r_ham = T_ham / ham_test
    r_spam = T_spam / spam_test
    f1_ham = 2 * p_ham * r_ham / (p_ham + r_ham)
    f1_spam = 2 * p_spam * r_spam / (p_spam + r_spam)
    return  accuracy,p_ham,r_ham,f1_ham,p_spam,r_spam,f1_spam



if __name__ == '__main__':
    # Input Training data
    ham_train = 1000
    spam_train = 997
    ham_test = 400
    spam_test =400

    result4=np.load("result4.npy")
    fig, ax = plt.subplots()
    # ax.plot(result4[:, 0], result4[:, 1], 'ro-', label='Accuracy')
    ax.plot(result4[:, 0], result4[:, 2], 'bp-', label='Precision_ham')
    ax.plot(result4[:, 0], result4[:, 3], 'g^-', label='Recall_ham')
    ax.plot(result4[:, 0], result4[:, 4], 'c+-', label='F1_ham')
    ax.plot(result4[:, 0], result4[:, 5], 'mp-', label='Precision_spam')
    ax.plot(result4[:, 0], result4[:, 6], 'y^-', label='Recall_spam')
    ax.plot(result4[:, 0], result4[:, 7], 'k+-', label='F1_spam')
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

    plt.xlabel('Length of vocabulary')
    plt.ylabel('Performance of testing')
    plt.show()


    train_path = "D:/Projects/AI/project2/train/"
    filenames_ham=["train-ham-{}.txt".format(str(x).zfill(5)) for x in range(1,ham_train+1)]
    filenames_spam=["train-spam-{}.txt".format(str(x).zfill(5)) for x in range(1,spam_train+1)]
    filename_sw="D:/Projects/AI/project2/stopwords.txt"

    test_path = "D:/Projects/AI/project2/test/"
    filenames_test_ham = ["test-ham-{}.txt".format(str(x).zfill(5)) for x in range(1, ham_test+1)]
    filenames_test_spam = ["test-spam-{}.txt".format(str(x).zfill(5)) for x in range(1, spam_test+1)]

    while True:
        expm = input("Input the No. of experiment need to implement(1,2,3, 4 or 5):")
        if expm not in ['1', '2', '3', '4', '5']:
            print("Wrong input,please try again. Ipnut the No. of experiment( 1, 2, 3, 4 or 5):")
        else:
            expm = int(expm)
            break

    vocabulary_ham=get_Vocabulary(filenames_ham)
    vocabulary_spam=get_Vocabulary(filenames_spam)

    alphabet=set(vocabulary_ham.keys()).union(vocabulary_spam.keys())

    ## Experiment 1: baseline#########################
    if expm==1:
        model_name = 'model.txt'
        result_name = 'baseline-result.txt'
    ## Experiment 2: stop_word filtering##############

    elif expm==2:
        stop_word=[]
        with open(filename_sw,'r') as file1:
            for line in file1.readlines():
                stop_word.append(line[:-1])
            file1.close()
        alphabet=[x for x in alphabet if x not in stop_word]
        model_name='stopword-model.txt'
        result_name='stopword-result.txt'

    ##Experiment 3:Word Length Filtering#############
    elif expm==3:
        alphabet = [x for x in alphabet if len(x)>2 and len(x)<9]
        model_name = 'wordlength-model.txt'
        result_name = 'wordlength-result.txt'

    if expm in [1,2,3]:
        vocabulary_list = sorted(alphabet, key=lambda item: item, reverse=False)
        print("vocabulary length:",len(vocabulary_list))
        voc_maxtix=get_PMatrix(vocabulary_list, vocabulary_ham, vocabulary_spam)

        with open('D:/Projects/AI/project2/'+model_name,'w',encoding='latin-1') as file:
            for i in range(len(vocabulary_list)):
                file.write("{}  {}  {}  {}  {}  {}\n".format(i + 1, vocabulary_list[i], int(voc_maxtix[i][0]), voc_maxtix[i][1], int(voc_maxtix[i][2]), voc_maxtix[i][3]))
            file.close()

        # Test the model with test data
        accuracy,p_ham, r_ham, f1_ham,p_spam, r_spam,f1_spam =Model_test([filenames_test_ham,filenames_test_spam], vocabulary_list, voc_maxtix, result_name, w=1)

        print("p_ham:{0:.3f},r_ham:{1:.3f},f1_ham:{2:.3f},p_spam:{3:.3f},r_spam:{4:.3f},f1_spam:{5:.3f}".format(p_ham, r_ham, f1_ham,p_spam, r_spam,f1_spam))

    # ##Experiment4: Infrequent Word Filtering########
    if expm==4:
        rmf = [1, 5, 10, 15, 20]        # Gradually remove words with frequency <=rmf[i] from the vocabulary
        rmf2= [5,10,15,20,25]           # Gradually remove top 5%, 10% ,15%,20% and 25%  most frequent words from the vocabulary
        result4=np.zeros((len(rmf) + len(rmf2), 8), dtype=float)   # length, accuarcy,p_ham,r_ham,f1_ham,p_spam,r_spam,f1_spam

        vocabulary1=vocabulary_ham.copy()
        for w in vocabulary_spam:
            if w in vocabulary1:
                vocabulary1[w]+=vocabulary_spam[w]
            else:
                vocabulary1[w] = vocabulary_spam[w]

        # vocabulary2=vocabulary1.copy()           # a copy of complte vocabulary for high frequent word removing experiment
        # raw_len=len(vocabulary1)                   # Length of the vocabulary without filtering
        for i in range(len(rmf)):
            rm=rmf[i]
            vocabulary1_list=list(vocabulary1.keys())
            for w in vocabulary1_list:
                if vocabulary1[w]<=rm:        # remove word with frequency <=rm
                    vocabulary1.pop(w)
            result4[i][0] = len(vocabulary1)
            vocabulary1_list = list(vocabulary1.keys())

            voc_maxtix = get_PMatrix(vocabulary1_list, vocabulary_ham, vocabulary_spam)
            result4[i, 1:] = Model_test([filenames_test_ham, filenames_test_spam], vocabulary1_list, voc_maxtix)

        # Removing high frequent word from vocabulary gradually.
        vocabulary2_list = sorted(vocabulary1.items(), key=lambda kv: kv[1], reverse=True)
        vocabulary2_list = [x[0] for x in vocabulary2_list]
        raw_len = len(vocabulary2_list)
        for i in range(len(rmf2)):
            rm=rmf2[i]
            num_cut=int(raw_len*rm/100)
            voc_list=vocabulary2_list[num_cut:]
            result4[i + len(rmf)][0]=len(voc_list)

            voc_maxtix = get_PMatrix(voc_list, vocabulary_ham, vocabulary_spam)

            result4[i + len(rmf), 1:]= Model_test([filenames_test_ham, filenames_test_spam], voc_list, voc_maxtix)
        np.save("result4.npy", result4)

        # Plot the result of experiment 4.
        fig, ax = plt.subplots()
        ax.plot(result4[:, 0], result4[:, 1], 'ro-', label='Accuracy')
        ax.plot(result4[:, 0], result4[:, 2], 'b*-', label='Precision_ham')
        ax.plot(result4[:, 0], result4[:, 3], 'b^-', label='Recall_ham')
        ax.plot(result4[:, 0], result4[:, 4], 'b+-', label='F1_ham')
        ax.plot(result4[:, 0], result4[:, 5], 'g*-', label='Precision_spam')
        ax.plot(result4[:, 0], result4[:, 6], 'g^-', label='Recall_spam')
        ax.plot(result4[:, 0], result4[:, 7], 'g+-', label='F1_spam')
        legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

        plt.xlabel('Length of vocabulary')
        plt.ylabel('Performance of testing')
        plt.show()

    if expm==5:
        result5=np.zeros((11, 8), dtype=float)   # length, accuarcy,p_ham,r_ham,f1_ham,p_spam,r_spam,f1_spam

        vocabulary_list=list(alphabet)
        i=0
        deltas=[x/10 for x in range(11)]
        for delta in deltas:
            voc_maxtix = get_PMatrix(vocabulary_list, vocabulary_ham, vocabulary_spam,sm=delta)
            # Test the model with test data
            result5[i][0] = delta
            result5[i, 1:]= Model_test([filenames_test_ham, filenames_test_spam], vocabulary_list, voc_maxtix)
            i+=1
        np.save("result5.npy", result5)
        # Plot the result of experiment 5.
        fig, ax = plt.subplots()
        ax.plot(result5[:, 0], result5[:, 1], 'ro-', label='Accuracy')
        ax.plot(result5[:, 0], result5[:, 2], 'b*-', label='Precision_ham')
        ax.plot(result5[:, 0], result5[:, 3], 'b^-', label='Recall_ham')
        ax.plot(result5[:, 0], result5[:, 4], 'b+-', label='F1_ham')
        ax.plot(result5[:, 0], result5[:, 5], 'g*-', label='Precision_spam')
        ax.plot(result5[:, 0], result5[:, 6], 'g^-', label='Recall_spam')
        ax.plot(result5[:, 0], result5[:, 7], 'g+-', label='F1_spam')
        legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

        plt.xlabel(r'$Smoothing value \delta$')
        plt.ylabel('Performance of testing')
        plt.show()
