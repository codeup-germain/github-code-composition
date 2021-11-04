import unicodedata
import re
import json

import pandas as pd
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split


################################## NLP ################################

def basic_clean(string):
    '''
    Inputs a string and lower cases it, normalized, and removed anything that is not a-z0-9
    '''
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string


def tokenize(string):
    '''
    Inputs a string and tokenizes it using ToktokTokenizer()
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)
    return string

def stem(string):
    '''
    Inputs a string an stems down every word
    '''
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(string):
    '''
    Inputs a string and lemmatizes every word
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(string):
    '''
    Inputs a string and removes all stop words 
    '''
    stopword_list = stopwords.words('english')
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    article_without_stopwords = ' '.join(filtered_words)
    return article_without_stopwords

def split(df):
    '''
    Inputs a DataFrame and splits the data into train, validate, and test.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)

    return train, validate, test