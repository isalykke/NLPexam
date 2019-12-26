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

hej = "msdfk===%?? sdckjlls dlkjn aall this is not... funny !#€%"


def text_cleaner(text):

    clean_text = re.sub(r'[\>\.\,\'\"\!\?\:\;\-\_\=\(\)\|\*\@\#\&\$\€\"\/\%\+]+','', text)

    return clean_text

test = text_cleaner(hej)