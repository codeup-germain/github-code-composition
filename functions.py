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
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)
    return string

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_stopwords(string):
    stopword_list = stopwords.words('english')
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    article_without_stopwords = ' '.join(filtered_words)
    return article_without_stopwords

def split(df):
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    return train, validate, test