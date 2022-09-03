#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import numpy as np
import pandas as pd
import glob
import unidecode
from english_words import english_words_set

with open('partial_fit_model_no_stop', 'rb') as files:
    l = pickle.load(files)
    
with open('count_vectorizer_no_stop', 'rb') as f:
    t = pickle.load(f)    

with open('label_encoder_no_stop', 'rb') as fil:
    v = pickle.load(fil)


# In[ ]:


df_all = pd.read_csv('codemix_data.csv')


# In[346]:


l.classes_


# In[379]:


def predictFind_codemix(text): 
    
    global english_words_set
    code_mix_check = []
    for token in text.lower().split():
        if token in list(english_words_set):
            code_mix_check.append('Yes')
            text = text.replace(token, '')
        
    global l, t, v
    x = t.transform([text]).toarray() 
    lang = l.predict(x) 
    lang_per = l.predict_proba(x) 
    lang = v.inverse_transform(lang)
    print(f'''The language is {str(lang[0])} with {round(float(lang_per.max()*100), 1)}% confidence \n\n''') 
    
    lang_names = list(v.inverse_transform(l.classes_)) 
    probas_list = []
    for i in lang_per:
        for j in i:
            probas_list.append(j)
    
    proba_table = pd.DataFrame({'Language': lang_names,
                        'Probability': probas_list}).sort_values(by='Probability',
                                                            ascending=False).reset_index(drop=True)
                
    if 'Yes' in code_mix_check and str(proba_table['Language'].loc[0])!='English':
        with open(f'''{proba_table['Language'].loc[0]}-English (code-mix).txt''', 'a') as file:
            file.write(f'{text}\n')
            file.close()  


# In[383]:


df_all.shape[0] #rows


# In[190]:


n_iter = 0
for text in df_all['sentence'].values:
    n_iter+=1
    print(n_iter)
    predictFind_codemix(text)

