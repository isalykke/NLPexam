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


############# LOOP OVER ALL MONTHS WITH DIFFERENT NUMBER OF TOPICS ################


num_lda_topics = [3, 5, 7, 10, 15, 20, 25]


col_names = [name for name in df.columns] #make a list of the col names 
col_names.append('topics') #append "topics" to that list
#new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month

#loop over each df (one pr month) and find topics

for num in range(len(num_lda_topics)):
    #CREATE A NEW DATAFRAME FOR EACH NUMBER OF TOPICS
    new_df = pd.DataFrame(columns = col_names) #create a new dataframe with same col names to have all topics pr month
    
    #loop over each df (one pr month) and find topics
    for df in df_list:

        lda = clean_n_lda(df, num_lda_topics[num]) #perform lda on each month
        print(len(df))
        print(num)

        #append the topics to the df
        df['topics'] = [lda.show_topics(num_lda_topics[num])] * len(df) #append the topics to the current df
        new_df = pd.concat([new_df, df]) #concatenate all the dfs with topics to one new df

    new_df.to_csv(f"lda_with_{num_lda_topics[num]}topics.csv")


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

3 strategies
- just pick the number
- sort out words that occur in over 25% of the sentences
- The 20% less frequent words include the most information 
the most frequently used 18% of words account for over 80% of word occurances 

'''


########################################################################################################################################################################
############################## FULL DATASET ANALYSIS ###################################################################################################################
##########################################################################################################################################################################

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
    and frequency[token] < 10
    and token not in stoplist]
    for text in process_sent
]


#finally, lemmantize the tokens
cleaned_sentences = [
    [lmtzr.lemmatize(word) for word in document if word not in stoplist]
    for document in process_sent
]



##########################################
############ TOPIC MODELLING - LDA #######
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
lda.get_term_topics("security")


##########################################################################################
############ PLOTTING OUTPUT ##############################################################
############################################################################################

#from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
'''
############# What is the Dominant topic and its percentage contribution in each document(sentence) ##############

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda, corpus=corpus1, texts=cleaned_sentences)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(20)

############# The most representative sentence for each topic ##############

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet


############# Frequency Distribution of Word Counts in Documents (sentences) ##############

#essentially this plot shows how long the sentences are on average

doc_lens = [len(d) for d in df_dominant_topic.Text]

import matplotlib.pyplot as plt
%matplotlib inline
%pylab inline 

# Plot
plt.figure(figsize=(4,2), dpi=160)
plt.hist(doc_lens, bins = 100, color='navy')
plt.text(100, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(100,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(100,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(100,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(100,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
plt.gca().set(xlim=(0, 7), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,7,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=10))
plt.show()


import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib
#matplotlib.use('Qt4Agg')
%pylab inline 
# the answer to all your problems https://stackoverflow.com/questions/7534453/matplotlib-does-not-show-my-drawings-although-i-call-pyplot-show
import matplotlib.pyplot as plt

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(2,2,figsize=(4,3.5), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 1000, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 1000), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i], fontsize=5)
    ax.set_title('Topic: '+str(i), fontdict=dict(size=8, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=10)
plt.show()
'''
############# Wordcloud of Top N words in each topic ##############

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stoplist,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=20,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda.show_topics(20, formatted=True)

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
plt.show()


############# Word Counts of Topic Keywords ##############

from collections import Counter
topics = lda_model.show_topics(20, formatted=False)
data_flat = [w for w_list in cleaned_sentences for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    #ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()

############# Sentence Chart Colored by Topic ##############
# Sentence Coloring of N Sentences
from matplotlib.patches import Rectangle

def sentences_chart(lda_model=lda, corpus=corpus1, start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1] 
            topic_percs, wordid_topics, wordid_phivalues = lda[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)       

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()

sentences_chart() 



############# t-SNE Clustering Chart ##############

# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda[corpus1]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)


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
### INTERACTIVE GRAPH OF LDA - PYLDAVIS ###
###########################################


# more info here https://pyldavis.readthedocs.io/en/latest/modules/API.html
import pyLDAvis.gensim

panel = pyLDAvis.gensim.prepare(lda, corpus1, dictionary, mds='TSNE')
pyLDAvis.display(panel)


##########################################
######### SOME PLOTTING - HEATMAP ##################
###########################################


from collections import OrderedDict
#set a range of topics found - in this case we set 5
data_hdp = {i: OrderedDict(lda.show_topic(i,20)) for i in range(6)}
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

#include all words that occur more than once dor word to vec

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



#make a word2vec model with the podcasts
from gensim.models import Word2Vec
security_now_model = Word2Vec(cleaned_sentences, size=100, window=5, min_count=3, workers=4, sg = 1)

print(security_now_model['girl'])
print(security_now_model.wv.most_similar('leash'))

#from https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html?fbclid=IwAR3aqPPGlCN5bNpN0lR_wR1nErLxCdPiksmlJIHmRtaqjzUa7Pjj7RNsoO0
from sklearn.decomposition import PCA
def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab), sample)
        else:
            words = [ word for word in model.vocabulary ]
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)


display_pca_scatterplot(security_now_model, 
                        ['uber', 'computer', 'bitcoin', 'crypto', 'security'])

display_pca_scatterplot(security_now_model, sample = )
