'''
Created on Nov 18, 2016

@author: miko
'''
import gensim
from nltk.tokenize.casual import TweetTokenizer
import word2vec

import numpy as np
import pandas as pd


dataset = pd.read_csv('Sentiment Analysis Dataset.csv', error_bad_lines=False)
dataset = dataset.drop(dataset.columns[0], 1)
dataset = dataset.drop(dataset.columns[1], 1)

"""
    bag of sentence
"""
text = dataset['SentimentText'].tolist() #make it a list
text = list(map(str.strip,text)) #remove any whitespace


dataset['SentimentText'] = text
cut = np.random.rand(len(dataset)) < 0.2
dataset = dataset[cut] #get 20% random data
tknzr = TweetTokenizer()
train_data = []
msk = np.random.rand(len(dataset)) < 0.6
train = dataset[msk]
test = dataset[~msk]
msk2 = np.random.rand(len(test)) < 0.5
validation = test[msk2]
test= test[~msk2]

for index, row in train.iterrows():
    train_data.append(tknzr.tokenize(row['SentimentText']))

model = gensim.models.Word2Vec(dataset['SentimentText'],max)
print model["I"]