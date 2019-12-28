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

df = pd.read_csv('preprocessed_txts_full_episodes.csv', encoding="utf8")

#find unique speakers 
x = np.array(list(df['speakers']))
len(np.unique(x))

episodes = list(df.whole_episode)
episodes

##########################################
############# TEXT CLEANUP ################
############################################

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


#make a list of lists of words
process_episode = [[word for word in document.lower().split()] for document in process_episode]

#calculate the frequency of each word, save it in a dictionary of format {"word":n}
import _collections
from _collections import defaultdict

frequency = defaultdict(int)
for text in process_episode:
    for token in text:
        frequency[token] += 1



#remove words that appear only once, as well as words in our stop list
#prepare the lemmatizer and the stop list
lmtzr = WordNetLemmatizer()
stoplist = stopwords.words('english')

process_episode = [
    [token for token in text if frequency[token] > 1 #and ordet optræder i over 25% af alle sætninger??
    and frequency[token] <= 10
    and token not in stoplist]
    for text in process_episode
]


#finally, lemmantize the tokens
cleaned_episodes = [
    [lmtzr.lemmatize(word) for word in document if word not in stoplist]
    for document in process_episode
]


##########################################
############ TOPIC MODELLING - LDA #######
############################################

import gensim
from gensim import corpora, models, similarities

#calculate frequencies
dictionary = corpora.Dictionary(cleaned_episodes) #a mapping between words and their integer ids.
corpus1 = [dictionary.doc2bow(episode) for episode in cleaned_episodes]


# Do the LDA - you must choose number of topics
total_topics = 100
lda_all = models.LdaMulticore(corpus1, id2word=dictionary, num_topics = total_topics, workers = 8)

lda_all.show_topics(total_topics,100)


##########################################
###### LATENT SEMENTIV INDEXING #######
############################################

#do an lsimodel - can estimate the optimal number of topics by itself

lsimodel = models.LsiModel(corpus1, num_topics=100, id2word=dictionary)
lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics
lsitopics = lsimodel.show_topics(formatted=False)


##########################################
###### HIERACHICAL DIRICHLET PROCESS #######
############################################

#do an HDP model - fully unsupervised - finds right number of topics
hdp = models.HdpModel(corpus1, id2word=dictionary)
hdp.print_topics()



'''
might be useful::
https://stackoverflow.com/questions/31543542/hierarchical-dirichlet-process-gensim-topic-number-independent-of-corpus-size/44393919#44393919

import pandas as pd
import numpy as np 

def topic_prob_extractor(hdp=None, topn=None):
    topic_list = hdp.show_topics(topics=-1, topn=topn)
    topics = [int(x.split(':')[0].split(' ')[1]) for x in topic_list]
    split_list = [x.split(' ') for x in topic_list]
    weights = []
    for lst in split_list:
        sub_list = []
        for entry in lst: 
            if '*' in entry: 
                sub_list.append(float(entry.split('*')[0]))
        weights.append(np.asarray(sub_list))
    sums = [np.sum(x) for x in weights]
    return pd.DataFrame({'topic_id' : topics, 'weight' : sums})
