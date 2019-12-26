##########################################
######### BONUS: WORD2VEC #################
###########################################
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
from matplotlib import pyplot as plt

security_now_model = Word2Vec(cleaned_sentences, size=100, window=5, min_count=3, workers=8, sg = 1)

print(security_now_model['girl'])
print(security_now_model.wv.most_similar('girl'))

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


display_pca_scatterplot(security_now_model, ['uber', 'computer', 'bitcoin', 'crypto', 'security'])

display_pca_scatterplot(security_now_model, sample = 100)