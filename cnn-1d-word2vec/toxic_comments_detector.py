#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:02:55 2018

@author: abhijeet
"""

from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from keras.models import model_from_json
import pandas as pd
import keras.backend as K
import pickle

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')


dataset = pd.read_csv("./test.csv", na_filter=False)
reviews = []

# Enconding Categorical Data     
cLen = len(dataset['comment_text'])
    
for i in range(0,cLen):
    review = dataset['comment_text'][i]
    reviews.append(review) 
   
    
tkr = RegexpTokenizer('[a-zA-Z0-9]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, review in enumerate(reviews):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(review)]
    tokenized_corpus.append(tokens)

print "tokenization done !!"
    
# Copy word vectors and delete Word2Vec model  and original corpus to save memory
pkl_file1 = open('word2vec.pkl', 'rb')
X_vecs = pickle.load(pkl_file1)

max_review_length = 100

indexes = set(np.random.choice(len(tokenized_corpus), len(reviews), replace=False))

del reviews
pred_scores = np.zeros((cLen,6))

for i, index in enumerate(indexes):
    X_test = np.zeros((1, max_review_length, 256), dtype=K.floatx())        
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_review_length:
            break
        
        if token not in X_vecs:
            continue
        X_test[0, t, :] = X_vecs[token]

    prediction = model.predict(X_test)
    pred_scores[i,0] = prediction[0][0]
    pred_scores[i,1] = prediction[0][1]
    pred_scores[i,2] = prediction[0][2]
    pred_scores[i,3] = prediction[0][3]
    pred_scores[i,4] = prediction[0][4]
    pred_scores[i,5] = prediction[0][5]
            










"""
#        X_test[i, t, :] = X_vecs[token]
       
del X_vecs

cLen = len(dataset['comment_text'])

prediction = model.predict(X_test)

pred_scores = np.zeros((cLen,6))

for pred in prediction:
    pred_scores[i,0] = pred[0][0]
    pred_scores[i,1] = pred[0][1]
    pred_scores[i,2] = pred[0][2]
    pred_scores[i,3] = pred[0][3]
    pred_scores[i,4] = pred[0][4]
    pred_scores[i,5] = pred[0][5]
"""    

raw_data = {'id': dataset['id'], 
        'toxic': pred_scores[:,0],
        'severe_toxic': pred_scores[:,1],
        'obscene': pred_scores[:,2],
        'threat': pred_scores[:,3],
        'insult': pred_scores[:,4],
        'identity_hate': pred_scores[:,5]}

df = pd.DataFrame(raw_data, columns = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
df.to_csv('submission_dnn.csv', sep=',',index=False)


