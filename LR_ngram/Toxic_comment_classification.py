#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:38:49 2018

@author: abhijeet
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

#Exploratory data analysis
#print train_df.sample(5)

cols_target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#print train_df.describe()    

#check if there is "null" comment

#print train_df[train_df['comment_text'].isnull()]
#print test_df[test_df['comment_text'].isnull()]

test_df.fillna('unknown',inplace=True)

print len(train_df), len(test_df)

print train_df[cols_target].sum()

print train_df['comment_text'].str.len().hist()

# finding the correlation among data
sns.set()
data = train_df[cols_target]
colormap = plt.cm.Reds
plt.figure(figsize=(5,5))
plt.title("Correlation of classes & Features")
sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax = 1.0, cmap=colormap, annot=True)
plt.show()
# Correlation says "insult & obscene | toxic & obscene | insult & toxic"

print test_df['comment_text'].str.len().hist()
plt.show()

print test_df[test_df['comment_text'].str.len() > 5000]


