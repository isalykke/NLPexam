import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import gensim
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel

import _collections
from _collections import defaultdict

import pandas as pd
import numpy as np 
import re, string, statistics

import tqdm
from tqdm

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

df = pd.read_csv('preprocessed_txts_full_episodes.csv', encoding="utf8")

#find unique speakers 
df['speakers'].head()
x = np.array(list(df['speakers']))
np.unique(x)
df.whole_episode

###########################################################################################################
############################## DEFINE FUNCTIONS + PREPARE DATA #############################################
############################################################################################################

#convert month and year to strings
df["month"] = df["month"].astype('str')
df["year"] = df["year"].astype('str')
#create unique month to split by
df['unique_month'] = df[['year', 'month']].apply('_'.join, axis=1)

#create a list containing datasets split by unique_month to loop over:
df_list = [df for _, df in df.groupby(df['episode'])]

#prepare stoplist and lemmatizer
stoplist = stopwords.words('english')
lmtzr = WordNetLemmatizer()



def word_cloud_func(lda):

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(stopwords=stoplist,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=20,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

    topics = lda.show_topics(20, formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()


#takes as input a dataframe and returns the same dataframe with a cleaned version of the episodes
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

'''
#we use the frequencies of informative words (e.g. windows: 11200) to determine the cutoffs for text cleaning
from collections import Counter 
k = Counter(frequency) 

#inspect values manually
high = k.most_common(100)
high

# Finding percentage values for cutoffs
procentage = 0.19
high = k.most_common(int(procentage*len(frequency)))
high[-1]
'''


#df = df_cleaner(df, 173)

def lda_maker(df, num_lda_topics):

    cleaned_episodes = [token.split() for token in df['clean_episode']]

    dictionary = corpora.Dictionary(cleaned_episodes) #a mapping between words and their integer ids.
    corpus1 = [dictionary.doc2bow(episode) for episode in cleaned_episodes]

    #corpus2 = dictionary.doc2bow(cleaned_episodes[2])
    #corpus1 = [dictionary.doc2bow(episode) for episode in cleaned_episodes]


    #create lda model and coherence model
    lda = models.LdaMulticore(corpus1, id2word = dictionary, num_topics = num_lda_topics, workers = 8)

    return lda



def coherence_maker(df, lda):

    cleaned_episodes = [token.split() for token in df['clean_episode']]

    dictionary = corpora.Dictionary(cleaned_episodes) #a mapping between words and their integer ids.
    corpus1 = [dictionary.doc2bow(episode) for episode in cleaned_episodes]

    lda_coherence = gensim.models.CoherenceModel(model = lda, texts = cleaned_episodes, dictionary = dictionary, coherence = 'c_v')

    return lda_coherence


###########################################################################################################
############################## LOOP OVER ALL EPISODES #############################################
############################################################################################################


num_lda_topics = [5] #set number of topics to loop over (min 4 for wordcloud)

col_names = [name for name in df.columns] #make a list of the col names 
col_names.append('clean_episode')
col_names.append('topics') #append "topics" to that list


#loop over each number of topics
for num in range(len(num_lda_topics)):

    new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month

    model_list = []
    coherence_values = []

    #loop over each df (one pr unique episode) and find topics
    for df in df_list:

        clean_df = df_cleaner(df, 19)
        lda = lda_maker(clean_df, num_lda_topics[num])
        lda_coherence = coherence_maker(clean_df, lda)s
        

        model_list.append(lda)
        coherence_values.append(lda_coherence.get_coherence())

        #wordcloud = word_cloud_func(lda)
        #plt.savefig(fname = f"wordclouds/word_cloud_for{df['unique_month'][0:1]}.png")

        #append the topics to the df
        df['topics'] = [lda.show_topics(num_lda_topics[num])] * len(df) #append the topics to the current df
        new_df = pd.concat([new_df, df]) #concatenate all the dfs with topics to one new df

    new_df.to_csv(f'lda_with_{num_lda_topics[num]}topics.csv')

    print(model_list)
    print(coherence_values)




