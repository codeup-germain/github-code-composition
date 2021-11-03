import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from prepare import prepare

import sklearn.preprocessing
import warnings
import re

from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
# imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

with open('data.json') as json_file:
    data = json.load(json_file)

df = pd.DataFrame(data)

train,validate,test = prepare(df)

def vectorizer_split(x):
    '''Function takes in X train, validate and test.
       Returns a count vectorized matrices for each.
       '''
    vectorizer = CountVectorizer(binary = True, stop_words = 'english')
    vectorizer.fit(list(train[x]))
    X_train = vectorizer.transform(train[x])
    X_validate= vectorizer.transform(validate[x])
    X_test = vectorizer.transform(test[x])
    return X_train.todense(),X_validate.todense(),X_test.todense()

def tfidf_split(x):
    '''Function takes in X train, validate and test.
    Returns a TFIDF vectorized matrices for each.
    '''
    tfidf = TfidfVectorizer()
    tfidf.fit(list(train[x]))
    X_train = tfidf.transform(train[x])
    X_validate= tfidf.transform(validate[x])
    X_test = tfidf.transform(test[x])
    return X_train.todense(),X_validate.todense(),X_test.todense()

def test_a_model(X_train, y_train, X_validate, y_validate, model, model_name, score_df):
    '''
    Function takes in X and y train
    X and y validate (or test) 
    A model with it's hyper parameters
    And a df to store the scores 
    - Set up an empty dataframe with score_df first
    - score_df = pd.DataFrame(columns = ['model_name', 'train_score', 'validate_score'])
    '''
    this_model = model

    this_model.fit(X_train, y_train)

    # Check with Validate

    train_score = this_model.score(X_train, y_train)
    
    validate_score = this_model.score(X_validate, y_validate)
    
    model_dict = {'model_name': model_name, 
                  'train_score': train_score, 
                  'validate_score':validate_score}
    score_df = score_df.append(model_dict, ignore_index = True)
    
    return score_df

def run_models():
    '''Function takes in X and y train
    X and y validate (or test) 
    And runs 20 models for random forrest, desciesion tree, and knn with varying hyperparametes
    This function then stores the models to a list of models
    - Set up with tree_models, forest_models,knn_models, tree_df, forest_df, knn_df = run_models()
    - pull the best performing model afterwards ex. tree_models[tree_df.validate_accuracy.idxmax()]'''
    #Decision Tree
    ## Create a for loop that creates 20 decision tree models with increasingly larger depths.
    metrics = []
    tree_models = []
    for i in range(2, 22):
        # Make the model
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)

        # Fit the model (on train and only train)
        tree = tree.fit(X_train, y_train)
        y_predictions = tree.predict(X_train)
        y_pred = tree.predict(X_validate)
        # Use the model
        in_sample_accuracy = round(tree.score(X_train, y_train),3)
    
        out_of_sample_accuracy = round(tree.score(X_validate, y_validate),3)
        
        in_sample_recall = round(sklearn.metrics.recall_score(y_train, y_predictions, pos_label=0, average='micro'),3)
        
        out_of_sample_recall = round(sklearn.metrics.recall_score(y_validate, y_pred, pos_label =0, average='micro'),3)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy,
            "train_recall": in_sample_recall,
            "validate_recall": out_of_sample_recall
        }
        
        # This creates the df below
        metrics.append(output)
        # tree_models will store all of my tree models incase i want them later
        tree_models.append(tree)
        
    tree_df = pd.DataFrame(metrics)
    tree_df["accuracy_difference"] = tree_df.train_accuracy - tree_df.validate_accuracy    
    
    # Random Forest
    ## Create a for loop that creates 20 Random Forrest models with increasingly larger depths.
    metrics2 = []
    forest_models = []
    for i in range(2, 22):
        # Make the model
        forest = RandomForestClassifier(max_depth=i, random_state=123)
    
        # Fit the model (on train and only train)
        forest = forest.fit(X_train, y_train)
        
        y_predictions = forest.predict(X_train)
        y_pred = forest.predict(X_validate)
        
        # Use the model
        in_sample_accuracy = round(forest.score(X_train, y_train),3)
        
        out_of_sample_accuracy = round(forest.score(X_validate, y_validate),3)
        
        in_sample_recall = round(sklearn.metrics.recall_score(y_train, y_predictions, pos_label =0, average='micro'),3)
        
        out_of_sample_recall = round(sklearn.metrics.recall_score(y_validate, y_pred, pos_label =0, average='micro'),3)
        
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy,
            "train_recall": in_sample_recall,
            "validate_recall": out_of_sample_recall
        }
        
        # This creates the df below
        metrics2.append(output)
        # tree_models will store all of my tree models incase i want them later
        forest_models.append(forest)
        
        
        
    forest_df = pd.DataFrame(metrics2)
    forest_df["accuracy_difference"] = forest_df.train_accuracy - forest_df.validate_accuracy
    

    knn_metrics = []
    knn_models = []
    # loop through different values of k
    for k in range(1, 21):
            
        # define the thing
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # fit the thing (remmeber only fit on training data)
        knn.fit(X_train, y_train)
        
        y_predictions = knn.predict(X_train)
        y_pred = knn.predict(X_validate)
        # use the thing (calculate accuracy)
        train_accuracy = round(knn.score(X_train, y_train),3)
        validate_accuracy = round(knn.score(X_validate, y_validate),3)
        train_recall = round(sklearn.metrics.recall_score(y_train, y_predictions, pos_label =0,average='micro'),3)
        validate_recall = round(sklearn.metrics.recall_score(y_validate, y_pred, pos_label =0,average='micro'),3)
        output = {
            "k": k,
            "train_accuracy": train_accuracy,
            "validate_accuracy": validate_accuracy,
            'train_recall':train_recall,
            "validate_recall":validate_recall
        }
        
        knn_metrics.append(output)
        knn_models.append(knn)
        # make a dataframe
    
    knn_df = pd.DataFrame(knn_metrics)
    knn_df["accuracy_difference"] = knn_df.train_accuracy - knn_df.validate_accuracy
        
    
    return tree_models, forest_models,knn_models, tree_df, forest_df, knn_df


def make_models_and_print_metrics_test_data(model, model_name, X_train, y_train, X_test, y_test, class_names):
    '''
    This function takes in a model object,
    Name for the model (for vis purposes)
    X_train, y_train
    X_test and y_test
    and the names of your classes (aka category names)
    Uses print metrics function 
    Use this function on the final test data set. 
    '''
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    
    print(f'                   ============== {model_name} ================           ')
    #print metrics for Test
    print_metrics(model, X_test, y_test, test_pred, class_names, set_name='Test')
    print('-------------------------------------------------------------------\n')
    
    
    
def print_metrics(model, X, y, pred, class_names, set_name = 'This Set'):
    '''
    This function takes in a model, 
    X dataframe
    y dataframe 
    predictions 
    Class_names (aka ['Java', 'Javascript', 'Jupyter Notebook', 'PHP'])
    and a set name (aka train, validate or test)
    Prints out a classification report 
    and confusion matrix as a heatmap
    To customize colors change insdie the function
    - IMPORTANT change lables inside this function
    '''
    
    
    print(model)
    print(f"~~~~~~~~{set_name} Scores~~~~~~~~~")
    print(classification_report(y, pred))
    
    #purple_cmap = sns.cubehelix_palette(as_cmap=True)
    purple_cmap = sns.color_palette("light:blue", as_cmap=True)
    
    with sns.axes_style("white"):
        matrix = plot_confusion_matrix(model,X, y, display_labels=class_names, 
                                       cmap = purple_cmap)
        plt.grid(False)
        plt.show()
        print()


