#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:35:50 2018

@author: abhijeet
"""

import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer 
from gensim.models.word2vec import Word2Vec
import multiprocessing
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

trainset = pd.read_csv("./train.csv")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
              
labels = trainset[label_cols].values
comments = trainset['comment_text']

tkr = RegexpTokenizer('[a-zA-Z0-9]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

# stem and tokenize
for i,comment in enumerate(comments):
    tokenized_corpus.append([stemmer.stem(t) for t in tkr.tokenize(comment)])

#pkl_file1 = open('word2vec.pkl', 'rb')
#X_vecs = pickle.load(pkl_file1)

    

#Word2Vec Tokenize
vector_size = 512
window_size = 10

word2vec = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())    

X_vecs = word2vec.wv

output = open('word2vec.pkl', 'wb')
pickle.dump(X_vecs, output)
output.close() 



# Compute average and max comment length
avg_length = 0.0
max_length = 0

for comment in tokenized_corpus:
    if len(comment) > max_length:
        max_length = len(comment)
    avg_length += float(len(comment))
    
print('Average review length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max review length: {}'.format(max_length))

max_comment_length = 120

# Create train and test sets
# Generate random indexes
indexes = set(np.random.choice(len(tokenized_corpus), len(comments), replace=False))

X_train = np.zeros((len(comments), max_comment_length, 512), dtype=K.floatx())
print X_train.shape

for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_comment_length:
            break
        
        if token not in X_vecs:
            continue
    
        if i < len(comments):
            X_train[i, t, :] = X_vecs[token]
            
            
del comments
del X_vecs

print "Training starts.."        
# Keras convolutional model
batch_size = 32
nb_epochs = 10

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_comment_length, 512)))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, labels,
          batch_size=batch_size,
          shuffle=True,
          epochs=nb_epochs,
          validation_split=0.1,
          callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')   
