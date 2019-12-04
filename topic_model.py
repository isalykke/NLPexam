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


####################################################################################
############################## ANALYSIS BY MONTH ###############################
######################################################################################


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


def clean_n_lda(df, num_lda_topics):

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
        and frequency[token] < 10
        and token not in stoplist]
        for text in process_sent
    ]

    cleaned_sentences = [
        [lmtzr.lemmatize(word) for word in document if word not in stoplist]
        for document in process_sent
    ]
    
    
    ############# LDA MODEL ################

    dictionary = corpora.Dictionary(cleaned_sentences) #a mapping between words and their integer ids.
    corpus1 = [dictionary.doc2bow(sent) for sent in cleaned_sentences]

    #create lda model
    lda = models.LdaMulticore(corpus1, id2word=dictionary, num_topics = num_lda_topics, workers = 8) #should I set eta??

    return lda

#test = clean_n_lda(df_list[100], 20) # test on one dataframe


############# LOOP OVER ALL MONTHS ################


num_lda_topics = 20

col_names = [name for name in df.columns] #make a list of the col names 
col_names.append('topics') #append "topics" to that list
new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month

#loop over each df (one pr month) and find topics
for df in df_list:

    lda = clean_n_lda(df, num_lda_topics) #perform lda on each month
    print(len(df))

    #append the topics to the df
    df['topics'] = [lda.show_topics(num_lda_topics)] * len(df) #append the topics to the current df
    new_df = pd.concat([new_df, df]) #concatenate all the dfs with topics to one new df


'''
questions:

NUM OF TOPICS:
how many topis? - do a test by looping over parts of the dataset 
and try out different number of topics to see if they differ a lot accross different splits of the data
- number of topics might be a thing to include in discussion??

CUTOFF
what's a theoretically sound cutoff for tokens? based on frequency in individual df maybe?
come up with a threshold for how many times words occur - 
more informative - ziphean phenomena to find the right size?
three different ways of doing the cutof - run it with each  and report the difference

- just pick the number
- sort out words that occur in over 25% of the sentences
- 

'''



####################################################################################
############################## FULL DATASET ANALYSIS ###############################
######################################################################################

sentences = list(df.utterance)
sentences

##########################################
############# TEXT CLEANUP ################
############################################

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
import _collections
from _collections import defaultdict

frequency = defaultdict(int)
for text in process_sent:
    for token in text:
        frequency[token] += 1


#remove words that appear only once, as well as words in our stop list
#prepare the lemmatizer and the stop list
lmtzr = WordNetLemmatizer()
stoplist = stopwords.words('english')

process_sent = [
    [token for token in text if frequency[token] > 1 #and ordet optræder i over 25% af alle sætninger??
    and token not in stoplist]
    for text in process_sent
]


#finally, lemmantize the tokens
cleaned_sentences = [
    [lmtzr.lemmatize(word) for word in document if word not in stoplist]
    for document in process_sent
]

type(cleaned_sentences)



##########################################
###### TOPIC MODELLING - LDA #######
############################################

import gensim
from gensim import corpora, models, similarities

#calculate frequencies
dictionary = corpora.Dictionary(cleaned_sentences) #a mapping between words and their integer ids.
corpus1 = [dictionary.doc2bow(sent) for sent in cleaned_sentences]


# Do the LDA - you must choose number of topics
total_topics = 20
lda = models.LdaMulticore(corpus1, id2word=dictionary, num_topics = total_topics, workers = 8)

lda.show_topics(total_topics,20)
lda.get_term_topics("password")


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


##########################################
######### INTERACTIVE GRAPH OF LDA ########
###########################################


# more info here https://pyldavis.readthedocs.io/en/latest/modules/API.html
import pyLDAvis.gensim

panel = pyLDAvis.gensim.prepare(lda, corpus1, dictionary, mds='TSNE')
pyLDAvis.display(panel)


##########################################
######### SOME PLOTTING ##################
###########################################


from collections import OrderedDict
#set a range of topics found - in this case we set 5
data_hdp = {i: OrderedDict(hdp.show_topic(i,20)) for i in range(6)}
df_hdp = pd.DataFrame(data_hdp)
df_hdp = df_hdp.fillna(0).T
print(df_hdp.shape)
df_hdp
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
g=sns.clustermap(df_hdp.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=1, figsize=(10, 12))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()



##########################################
######### BONUS: WORD2VEC #################
###########################################

#make a word2vec model with the podcasts
from gensim.models import Word2Vec
security_now_model = Word2Vec(cleaned_sentences, size=100, window=5, min_count=3, workers=4, sg = 1)

print(security_now_model['female'])
print(security_now_model.wv.most_similar('leash'))

