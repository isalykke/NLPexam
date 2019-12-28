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

data_all = pd.read_csv('preprocessed_txts_full_episodes.csv', encoding="utf8")

data_all = data_all[1:5]

clean_dfs_dict = {}

full_dataset_results = []

#prepare stoplist and lemmatizer
stoplist = stopwords.words('english')
lmtzr = WordNetLemmatizer()

##########################################
############# TEXT CLEANUP ################
############################################

def df_cleaner(df, cutoff):

    ############# TEXT CLEANUP ################

    episodes = list(df.whole_episode)
    # RE removing words with length <2 characters
    process_episode = list(filter(None, [re.sub(r'\b\w{1,3}\b','', x) for x in episodes])) 
    # RE removing URLs
    process_episode = list(filter(None, [re.sub(r'http\S+','', x) for x in process_episode]))
    # RE removing numbers
    process_episode = list(filter(None, [re.sub(r'\d+','', x) for x in process_episode]))
    # RE removing punctuation
    process_episode = list(filter(None, [re.sub(r'[\>\€\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\&\$\"\/\%\+]+','', x) for x in process_episode]))
    # RE removing non-ASCII characters
    process_episode = list(filter(None, [re.sub(r'[^\x00-\x7F]+','', x) for x in process_episode]))

    #make a list of lists of utterances
    process_episode = [[word for word in document.lower().split()] for document in process_episode]

    #calculate the frequency of each word, save it in a dictionary of format {"word":n}
    frequency = defaultdict(int)
    no_total_words = 0

    for text in process_episode:
        for token in text:
            frequency[token] += 1
            no_total_words += 1

    #remove words that appear only once, more than cutoff, as well as words in our stop list
    process_episode = [
        [token for token in text if frequency[token] > 1
        and frequency[token] <= cutoff 
        and token not in stoplist]
        for text in process_episode
    ]

    cleaned_episodes = [
        [lmtzr.lemmatize(word) for word in document if word not in stoplist]
        for document in process_episode
    ]

    df['clean_episode'] = [" ".join(episode) for episode in cleaned_episodes]

    return df

def lda_maker(num_lda_topics, dictionary, corpus):

    #create lda model 
    lda = models.LdaMulticore(corpus1, id2word = dictionary, num_topics = num_lda_topics, workers = 8)

    return lda

def coherence_maker(lda, dictionary, clean_episodes):

    #create coherence model
    lda_coherence = gensim.models.CoherenceModel(model = lda, texts = cleaned_episodes, dictionary = dictionary, coherence = 'c_v')

    return lda_coherence


##########################################
############ LOOP OVER PARAMETERS #######
############################################

import gensim
from gensim import corpora, models, similarities

num_lda_topics = [20, 30] #set number of topics to loop over (min 4 for wordcloud)

cutoffs = [10, 56]


for num in range(len(num_lda_topics)):

    for cut in range(len(cutoffs)):

        #clean the dataframe once and create a dictionary+corpus for lda
        if cut in clean_dfs_dict:

            clean_df = clean_dfs_dict[cut]

        else:
            clean_df = df_cleaner(data_all, cutoffs[cut])
            
            clean_dfs_dict[cutoffs[cut]] = clean_df
    
        cleaned_episodes = [token.split() for token in clean_df['clean_episode']] #we need to split the cleaned words into a list for lda

        dictionary = corpora.Dictionary(cleaned_episodes) #a mapping between words and their integer ids.
        corpus1 = [dictionary.doc2bow(episode) for episode in cleaned_episodes]

        #perform the lda model and calculate coherence scores
        lda = lda_maker(num_lda_topics[num], dictionary, corpus1)
    
        lda_coherence = coherence_maker(lda, dictionary, cleaned_episodes)

        perplexity = pow(2, -(lda.log_perplexity(corpus1))) 

        #create a tupple with outcomes
        lda_stats = (num_lda_topics[num], cutoffs[cut], lda_coherence.get_coherence(), perplexity, lda)
        full_dataset_results.append(lda_stats)

        print(f'topics:{lda_stats[0]}, cutoff: {lda_stats[1]}, coherence: {lda_stats[2]}')


