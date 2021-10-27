import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def basic_clean(article):
    """ lowercases, normalizes, and destroys special characters of an article (string) """
    # lowercase
    article = article.lower()
    # normalize
    article = unicodedata.normalize('NFKD', article).encode('ascii', 'ignore').decode('utf-8')
    # remove special characters
    article = re.sub(r"[^a-z0-9'\s]", "", article)
    
    return article

def tokenize(article):
    """ tokenize a basic_clean-ed article (string) """
    tokenizer = nltk.tokenize.ToktokTokenizer() # create tokenizer
    article = tokenizer.tokenize(article, return_str = True) # tokenize
    
    return article

def stem(article):
    """ stem all words in an article (string) """
    ps = nltk.porter.PorterStemmer() # create stemmer
    stems = [ps.stem(word) for word in article.split()] # list comprehension of stems
    article_stemmed = ' '.join(stems) # re-join list as article
    
    return article_stemmed

def lemmatize(article):
    """ lemma all words in an article (string) """
    nltk.download('wordnet') # get current lemma list
    wnl = nltk.stem.WordNetLemmatizer() # create lemmatizer
    lemmas = [wnl.lemmatize(word) for word in article.split()] # list comp of lemmas
    article_lemmatized = ' '.join(lemmas) # re-join list as article
    
    return article_lemmatized

def remove_stopwords(article):
    """ remove stopwords from an article (string) """
    stopword_list = stopwords.words('english') # get default stopword list
    words = article_lemmatized.split() # split for stopword removal
    filtered_words = [word for word in words if word not in stopword_list] # ignore stopwords
    article_without_stopwords = ' '.join(filtered_words) # re-join list to article
    
    return article_without_stopwords