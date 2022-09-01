#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import stopwordsiso
import unidecode
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
cv = CountVectorizer(ngram_range=(1,1))


# In[250]:


df = pd.read_csv('11lang13million.csv') 


# In[251]:


def preProcessing(data: pd.DataFrame):
    
    '''Returns X and y'''
    
    X = data["Text"]
    y = data["Language"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    with open('label_encoder_indian', 'wb') as files:
        pickle.dump(le, files)
    
    data_list = []
    
    for text in X:
        text = re.sub(r'[!@#$(),n"%^*.?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        data_list.append(text)
    
    return X, y


# In[252]:


X, y = preProcessing(df)


# In[253]:


print('--- Preprocessing is done! ---')


# In[254]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 00.00075, stratify=y)


# In[255]:


print('--- Train-test-split is done! ---')


# In[256]:


x_train.reset_index(drop=True, inplace=True)
#y_train.reset_index(drop=True, inplace=True)


# In[257]:


def partialFit_predict(X, y, n_batches: int, n_iter: int):
    
    def batches(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    
    print('--- Batch func done! ---')
    
    global cv
    b = 0
    i = 0
    for _ in range(n_iter):
        i = i + 1 
        shuffled_data = pd.concat([X, pd.Series(y)], axis=1).rename(columns= {0: 'Lang'}).sample(frac=1)
        shuffledX = shuffled_data['Text']
        shuffledY = shuffled_data['Lang']
        print(f'--- Shuffle {i} --- \n\n')
        for batch in batches(range(len(X)), n_batches):
            model = MultinomialNB()
        
            b = b + 1
            print(f'Iteration {i}, Batch size {b}\n\n')
        
            try:
                x = cv.fit_transform(shuffledX[batch[0]:batch[-1]+1]).toarray()
                model.partial_fit(x, shuffledY[batch[0]:batch[-1]+1], classes=np.unique(y))
            except:
                x = cv.transform(shuffledX[batch[0]:batch[-1]+1]).toarray()
                model.partial_fit(x, shuffledY[batch[0]:batch[-1]+1], classes=np.unique(y))
    
    print('--- Training is done! ---\n\n')
    global y_test, x_test
    x_test = cv.transform(x_test).toarray()
    
    with open('count_vectorizer_indian', 'wb') as f:
        pickle.dump(cv, f)
    
    y_pred = model.predict(x_test)
    
    with open('model_indian', 'wb') as file:
        pickle.dump(model, file)
    
    ac = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(cr)


# In[258]:


partialFit_predict(x_train, y_train, 10000, n_iter=1)

