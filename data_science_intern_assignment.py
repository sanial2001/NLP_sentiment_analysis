#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import requests


# In[2]:


df = pd.read_csv('cik_list.csv')
df.head(4)


# In[5]:


i=0
list_of_data = []
while i < len(df):
    url = 'https://www.sec.gov/Archives/' + df.iloc[i, -1]
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    body = soup.body.text
    list_of_data.append(body)
    list_of_data = [x.replace('\n', ' ') for x in list_of_data]
    i = i + 1


# In[9]:


data = pd.DataFrame(list_of_data, columns=['SEC/EDGAR'])
data.head(5)


# In[12]:


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['SEC/EDGAR'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[14]:


corpus


# In[86]:


output_df = pd.read_csv('Output Data Structure.csv')
output_df


# In[87]:


uncertain_df = pd.read_csv('uncertainty_dictionary.csv')
constrain_df = pd.read_csv('constraining_dictionary.csv')
uncertain_list = uncertain_df['Word'].to_list()
constrain_list = constrain_df['Word'].to_list()


# In[88]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


# In[89]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
complex_words = 0
uncertain_count = 0
constrain_count = 0

for sentiment_text in corpus:
    length = len(sentiment_text.split())
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg'] * length / 100
    pos = score['pos'] * length / 100
    polarity = (pos-neg)/((pos+neg) + 0.000001) 
    pos_prop = int(pos) / length
    neg_prop = int(neg) / length
    
    words = sentiment_text.split()
    for i in words:
        if syllable_count(i) > 2:
            complex_words = complex_words + 1
        if any(word in i for word in uncertain_list):
            uncertain_count = uncertain_count + 1
        if any(word in i for word in constrain_list):
            constrain_count = constrain_count + 1
    uncertain_prop = uncertain_count / length
    constrain_prop = constrain_count / length
    complex_words_percent = complex_words / len(sentiment_text.split()) * 100
    average = sum(len(word) for word in words)/len(words)
    fog = 0.4 * (average + complex_words_percent)
    
    add_value = {'positive_score' : pos, 'negative_score' : neg, 'polarity_score' : polarity, 
                 'average_sentence_length' : average, 'percentage_of_complex_words' : complex_words_percent,
                 'fog_index' : fog, 'complex_word_count' : complex_words, 'word_count' : length,
                 'uncertainty_score' : uncertain_count, 'constraining_score' : constrain_count,
                 'positive_word_proportion' : pos_prop, 'negative_word_proportion' : neg_prop,
                 'uncertainty_word_proportion' : uncertain_prop, 'constraining_word_proportion' : constrain_prop,
                 'constraining_words_whole_report' : constrain_count}
    output_df = output_df.append(add_value, ignore_index=True)


# In[90]:


output_df


# In[91]:


output_df.to_csv('result.csv')

