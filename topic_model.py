import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np 
import re,string

df = pd.read_csv('preprocessed_txts.csv', encoding="utf8")
df.shape

#find unique speakers 
df['speakers'].head()
x = np.array(list(df['speakers']))
np.unique(x) 

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
    [token for token in text if frequency[token] > 1
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
total_topics = 5
lda = models.LdaModel(corpus1, id2word=dictionary, num_topics = total_topics)

lda.show_topics(total_topics,10)




#come up with a threshold for how many times words occur - more informative - ziphean phenomena to find the right size?
#strip out some more stopwords maybe
#split into dataframes in panda by month and do one lda model pr dataframe

#do a model for each month to 3 months and look at the top 5-6 topics in a loop
#save it in a dictionary with dataframe as the key and the topics as the value 


how to use a cleaned corpus for this?
for period in periods:
    create the corpus
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics = total_topics)


##########################################
###### LATENT SEMENTIV INDEXING #######
############################################

#do an lsimodel - can estimate the optimal number of topics by itself

lsimodel = models.LsiModel(corpus1, num_topics=20, id2word=dictionary)
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

