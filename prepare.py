from functions import basic_clean, tokenize, stem, lemmatize, remove_stopwords, split

def prepare(df):
    '''
    This prepare functions takes in a DataFrame and returns a DataFrame with a clean, stemmed,
    and lemmatized columns along with its original contents.  
    '''

    # Clean will take the readme_contents feature and apply a basic_clean, tokenize, and remove stopwords
    # functions on it.  
    df['clean'] = df.readme_contents.apply(basic_clean).apply(tokenize).apply(remove_stopwords)

    # Clean will take the readme_contents feature and apply a stem function on it
    df['stemmed'] = df['clean'].apply(stem)

    # Clean will take the readme_contents feature and apply a lemmatized on it 
    df['lemmatized'] = df['clean'].apply(lemmatize)

    # Split the data
    train, validate, test = split(df)

    return train, validate, test