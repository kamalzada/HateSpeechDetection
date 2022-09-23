#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import pickle
from glob import glob
import re
import nltk
import unidecode

with open('NB_Model', 'rb') as files:
    model = pickle.load(files)
    
with open('tfidf_vectorizer', 'rb') as f:
    tfidf = pickle.load(f)    

with open('Label_Encoder', 'rb') as fil:
    le = pickle.load(fil)


# In[12]:


le.inverse_transform(model.classes_)


# In[5]:


vocab = open('eng_vocab.txt')
voc = [str(word.replace('\n', '')) for word in vocab.readlines()]


# In[6]:


voc = [word for word in voc if word!='are' and word!='will' and word!='bill' and word!='a' and word!='me']


# In[7]:


vocab.close()


# In[5]:


#stop = stopwords.words('english')


# In[50]:


# def expand_contractions(text):
#     expanded_text = contractions.fix(text) 
#     return expanded_text


# In[51]:


# contracted_stop = []
# for i in stop:
#     contracted_stop.append(expand_contractions(i))


# In[52]:


# contracted_stw = []
# for i in contracted_stop:
#     contracted_stw.append(i)


# In[53]:


# all_stops = stop + contracted_stw


# In[54]:


# len(all_stops) 


# In[55]:


# from bs4 import BeautifulSoup
# import requests


# In[56]:


# page = requests.get('https://www.ef-australia.com.au/english-resources/english-vocabulary/top-3000-words/')
# soup = BeautifulSoup(page.text, 'lxml')


# In[57]:


# eng_words = []
# for i in soup.findAll('p')[11:]:
#     for j in i:
#         if j!=' ' and j!='':
#             eng_words.append(j.text)


# In[58]:


# eng_words = eng_words + all_stops


# In[59]:


# len(eng_words)


# In[60]:


# english_s = []


# In[61]:


# for word in eng_words:
#     english_s.append(f'{word}ing')


# In[62]:


# for word in eng_words:
#     english_s.append(f'{word}s')


# In[63]:


# for word in eng_words:
#     english_s.append(f'{word}es')


# In[64]:


# for word in eng_words:
#     english_s.append(f'{word}ed')


# In[65]:


# eng_words = english_s + eng_words


# In[66]:


# len(eng_words)


# In[ ]:


# import enchant
# d = enchant.Dict("en_US")
# d.check("Hello")

# eng_words = [word for word in eng_words if word!='']
# eng_words = [word for word in eng_words if d.check(word)==True] 
# len(eng_words)


# In[5]:


# vocab = open('eng_vocab.txt')
# vocab_list = [word.replace('\n', '') for word in vocab.readlines()]
# vocab.close()


# In[8]:


def predictFind_codemix(text): 
    
    global voc
    text = text.lower()
    text = re.sub(r'[!@#$(),"%^*.?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = re.sub(r' +', ' ', text)  
    english_words = []
    text_for_pred = []
    for token in text.split():
        if token in voc:
            english_words.append(token)
    
    text_for_pred = ' '.join([token for token in text.split() if token not in english_words])
    print(text_for_pred)
    global model, tfidf, le
    x = tfidf.transform([text_for_pred]).toarray() 
    lang = model.predict(x) 
    lang_per = model.predict_proba(x) 
    lang = le.inverse_transform(lang)
    print(f'''The text is in {str(lang[0])}\n\n''') 
    
    lang_names = list(le.inverse_transform(model.classes_)) 
    probas_list = []
    for i in lang_per:
        for j in i:
            probas_list.append(j)

    proba_table = pd.DataFrame({'Language': lang_names,
                        'Probability': probas_list}).sort_values(by='Probability',
                                                           ascending=False).reset_index(drop=True)

    if len(english_words)>0 and len(english_words)/len(text.split())<0.75:
        with open(f'''/media/Filtered_data_with_96%/{str(lang[0])}-English-codemix.csv''', 'a') as file:
            file.write(f'{text}\n')      
            file.close()
    elif len(english_words)==0:
        pass


# In[9]:


predictFind_codemix('tama bhauni mo senior thele school re ravensha girl school semane jn bhari sanga barsa apa satbdhi apa')


# In[15]:


files = glob('*.csv')
for index, file in enumerate(files):
    print(f'CSV file number: {index}')
    with open(file, 'r') as f:
        data = f.readlines()
f.close()        
for text in data:
    predictFind_codemix(text)


# In[ ]:


# x = tfidf.transform(['didi tumi vidio dite etoo derii korecho kenoo koto opekhai thaki tomar vidior di didi vidioo dio']).toarray() 
# lang = model.predict(x) 
# lang_per = model.predict_proba(x) 
# lang = le.inverse_transform(lang)
# print(f'''{str(lang[0])}\n\n''') 


# In[24]:


#deal with the duplicates later


# In[ ]:


# for index, file in enumerate(files):
#     print(f'CSV file number: {index}')
#     df = pd.read_csv(file, on_bad_lines='skip')
#     df.drop_duplicates(inplace=True)
#     for text in df.values:
#         predictFind_codemix(text)

