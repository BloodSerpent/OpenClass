'''
Created on Nov 18, 2016

@author: miko
'''
import nltk
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.util import *
from nltk.tokenize import TweetTokenizer
import pickle

import numpy as np
import pandas as pd

import sys
reload(sys)
sys.setdefaultencoding('utf8')

dataset = pd.read_csv('Sentiment Analysis Dataset.csv', error_bad_lines=False)
dataset = dataset.drop(dataset.columns[0], 1)
dataset = dataset.drop(dataset.columns[1], 1)

"""
    bag of sentence
"""
text = dataset['SentimentText'].tolist() #make it a list
text = list(map(str.strip,text)) #remove any whitespace


dataset['SentimentText'] = text
cut = np.random.rand(len(dataset)) < 0.1
dataset = dataset[cut] #get 20% random data
dataset_frame = []
# dataset_frame = pickle.load(open("dataset.pkl","rb"))
tknzr = TweetTokenizer()

for index, row in dataset.iterrows():
    dataset_frame.append((tknzr.tokenize(row['SentimentText']),row['Sentiment']))
 
with open('dataset.pkl','w') as f:
    pickle.dump(dataset_frame, f)
   
msk = np.random.rand(len(dataset)) < 0.6

train = dataset[msk]
test = dataset[~msk]

msk2 = np.random.rand(len(test)) < 0.5

validation = test[msk2]
test= test[~msk2]

#make modelnp.random.rand(len(dataset)) < 0.6


train_data = []
validation_data = []
test_data = []
 
for index, row in train.iterrows():
    train_data.append((tknzr.tokenize(row['SentimentText']),row['Sentiment']))
 
for index, row in validation.iterrows():
    validation_data.append((tknzr.tokenize(row['SentimentText']),row['Sentiment']))
 
for index, row in test.iterrows():
    test_data.append((tknzr.tokenize(row['SentimentText']),row['Sentiment']))  

# dataset preparation

 
print train_data[0]
print validation_data[0]
print test_data[0]

# with open('traind_data.pkl','w') as f:
#     pickle.dump(train_data, f)
#  
# with open('validation_data.pkl','w') as f:
#     pickle.dump(validation_data, f)
#      
# with open('test_data.pkl','w') as f:
#     pickle.dump(test_data, f)

# train_data = pickle.load(open("train_data.pkl","rb"))
# validation_data = pickle.load(open("validation_data.pkl","rb"))
# test_data = pickle.load(open("test_data.pkl","rb"))


# Naive Bayes Trainer
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in dataset_frame])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=0)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

train_data = sentim_analyzer.apply_features(train_data)
validation_data = sentim_analyzer.apply_features(validation_data)
test_data = sentim_analyzer.apply_features(test_data)
  
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, train_data)
for key,value in sorted(sentim_analyzer.evaluate(test_data).items()):
    print('{0}: {1}'.format(key, value))





