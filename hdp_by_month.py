import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
from gensim import corpora, models, similarities

import _collections
from _collections import defaultdict

import pandas as pd
import numpy as np 
import re,string

df = pd.read_csv('preprocessed_txts.csv', encoding="utf8")

#find unique speakers 
df['speakers'].head()
x = np.array(list(df['speakers']))
np.unique(x) 

df.head


########################################################################################################################################################################
############################## ANALYSIS BY MONTH ###################################################################################################################
##########################################################################################################################################################################


#convert month and year to strings
df["month"] = df["month"].astype('str')
df["year"] = df["year"].astype('str')
#create unique month to split by
df['unique_month'] = df[['year', 'month']].apply('_'.join, axis=1)


#create a list containing datasets split by unique_month to loop over:
df_list = [df for _, df in df.groupby(df['unique_month'])]


#prepare stoplist and lemmatizer
stoplist = stopwords.words('english')
lmtzr = WordNetLemmatizer()


def clean_n_lda(df):

    ############# TEXT CLEANUP ################

    sentences = list(df.utterance)
    # RE removing words with length <2 characters
    process_sent = list(filter(None, [re.sub(r'\b\w{1,3}\b','', x) for x in sentences])) 
    # RE removing URLs
    process_sent = list(filter(None, [re.sub(r'http\S+','', x) for x in process_sent]))
    # RE removing numbers
    process_sent = list(filter(None, [re.sub(r'\d+','', x) for x in process_sent]))
    # RE removing punctuation
    process_sent = list(filter(None, [re.sub(r'[\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\&\$\"\/\%\+]+','', x) for x in process_sent]))
    # RE removing non-ASCII characters
    process_sent = list(filter(None, [re.sub(r'[^\x00-\x7F]+','', x) for x in process_sent]))

    #make a list of lists of utterances
    process_sent = [[word for word in document.lower().split()] for document in process_sent]

    #calculate the frequency of each word, save it in a dictionary of format {"word":n}
    frequency = defaultdict(int)
    for text in process_sent:
        for token in text:
            frequency[token] += 1

    #remove words that appear only once, as well as words in our stop list
    process_sent = [
        [token for token in text if frequency[token] > 1 #change this number to only include more informative words
        #and frequency[token] < 10
        and token not in stoplist]
        for text in process_sent
    ]

    cleaned_sentences = [
        [lmtzr.lemmatize(word) for word in document if word not in stoplist]
        for document in process_sent
    ]
    
    
    ############# HDP MODEL ################

    dictionary = corpora.Dictionary(cleaned_sentences) #a mapping between words and their integer ids.
    corpus1 = [dictionary.doc2bow(sent) for sent in cleaned_sentences]

    #create hdp model
    hdp = models.HdpModel(corpus1, id2word=dictionary) #should I set eta??

    return hdp

#test = clean_n_lda(df_list[100], 20) # test on one dataframe


############# LOOP OVER ALL MONTHS ################


col_names = [name for name in df.columns] #make a list of the col names 
col_names.append('topics') #append "topics" to that list
new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month

#loop over each df (one pr month) and find topics
for df in df_list:

    hdp = clean_n_lda(df) #perform lda on each month
    print(len(df))

    #append the topics to the df
    df['topics'] = [hdp.show_topics] * len(df) #append the topics to the current df
    new_df = pd.concat([new_df, df]) #concatenate all the dfs with topics to one new df


new_df.to_csv('df_with_topics_hdp.csv')