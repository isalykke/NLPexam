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
df_list = [df for _, df in df.groupby(df['unique_month'])]

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

def clean_n_lda(df, num_lda_topics, cutoff):

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
    for text in process_episode:
        for token in text:
            frequency[token] += 1

    #remove words that appear only once, as well as words in our stop list
    process_episode = [
        [token for token in text if frequency[token] > 1 #change this number to only include more informative words
        and frequency[token] < cutoff
        and token not in stoplist]
        for text in process_episode
    ]

    cleaned_sentences = [
        [lmtzr.lemmatize(word) for word in document if word not in stoplist]
        for document in process_episode
    ]
    
    
    ############# LDA MODEL ################

    dictionary = corpora.Dictionary(cleaned_sentences) #a mapping between words and their integer ids.
    corpus1 = [dictionary.doc2bow(sent) for sent in cleaned_sentences]

    #create lda model
    lda = models.LdaMulticore(corpus1, id2word=dictionary, num_topics = num_lda_topics, workers = 8)

    return lda

test = clean_n_lda(df_list[100], 20, 10) # test on one dataframe


###########################################################################################################
############################## LOOP OVER ALL MONTHS #############################################
############################################################################################################


num_lda_topics = [20] #set number of topics to loop over (min 4 for wordcloud)

col_names = [name for name in df.columns] #make a list of the col names 
col_names.append('topics') #append "topics" to that list


#loop over each number of topics
for num in range(len(num_lda_topics)):

    new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month

    #loop over each df (one pr unique month) and find topics
    for df in df_list:

        lda = clean_n_lda(df, num_lda_topics[num], 10) #perform lda on each month
        print(len(df))
        #wordcloud = word_cloud_func(lda)
        #plt.savefig(fname = f"wordclouds/word_cloud_for{df['unique_month'][0:1]}.png")

        #append the topics to the df
        df['topics'] = [lda.show_topics(num_lda_topics[num])] * len(df) #append the topics to the current df
        new_df = pd.concat([new_df, df]) #concatenate all the dfs with topics to one new df

    new_df.to_csv(f'lda_with_{num_lda_topics[num]}topics.csv')
