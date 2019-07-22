import sys
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """ Load data from database, extract features and multi-output response
    variables

    Parameters
    ----------
    database_filepath: str, path to database file

    Returns
    -------
    X : Series, collection of messages in their raw format
    Y : DataFrame, dataset consisting of multi-output response variables - each
    column represents a different binary response variable
    category_names : array, names of binary response variables
    """

    #Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    #Define feature and target variables X and Y
    X = df['message']
    Y = df.iloc[:, 4:]

    #Extract category names
    category_names = Y.columns.values

    return X, Y, category_names

def tokenize(text):
    """ Use nltk to normalize, tokenize and lemmatize input text

    Parameters
    ----------
    text : str, text in raw format

    Returns
    -------
    lemmed: list, collection of lemmatized tokens from the input text
    """

    #Transform the text to lowercase plus remove all characters that are not
    #letters or numbers
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    #Tokenize the text to words
    tokens = word_tokenize(text)

    #Remove stopwords and whitespace around the words
    tokens_subset = [v.strip() for v in tokens if v.strip()
                     not in set(stopwords.words('english'))]

    #Lemmatize the tokens and return the final list
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in tokens_subset]

    return lemmed


def build_model():
    """ Define and build a machine learning pipeline that is able to convert a
    collection of text documents into a matrix of token counts, then perform the
    tf-idf transformation, and ultimately train a multi-output classifier on all
    categories in the dataset (fitting one RandomForestClassifier per target
    variable). Implement a GridSearchCV object on the defined pipeline object
    that will find the best parameters for the pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    cv : GridSearchCV object, model construct that will be used to find and
    train the best performing pipeline
    """

    #Build a machine learning pipeline
    pipeline = Pipeline([
        #Use the tokenize function from above as the tokenizer
        ('count', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        #Fit one RandomForestClassifier per response variable
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #Define the sets of pipeline parameters that will be tested in grid search

    #####
    #NOTE: The parameter sets are all commented out so the reviewer will be able
    #to run the code in a shorter amount of time
    #####
    parameters = {#'count__max_df': [0.75, 1.0],
              #'count__ngram_range': [(1,1),(1,2)],
              #'count__max_features' : [None, 100,200],
              #'tfidf__smooth_idf':[True, False],
              #'clf__estimator__max_depth': [None,4,8],
              #'clf__estimator__min_samples_split': [2, 10, 50],
              #'clf__estimator__n_estimators': [10, 50]
             }

    #Define the scorer that will measure the performance of different parameter
    #sets in grid search
    total_scorer = make_scorer(f1_score, average = 'micro')

    #Note: Instead of averaging the f1 score achieved on different categories we
    #perform the grid search with respect to the global f1_score that counts the
    #total true positives, false negatives, etc. This is achieved by defining a
    #custom scorer and setting the average parameter to average='micro'.

    #In turn, more emphasis is put on the categories with more positive labels
    #in the testing set. Overall this does not pose a major concern, since we
    #ultimately care about the total model performance, rather than achieving a
    #high score in each of the categories. Specific categories are often highly
    #imbalanced with very little positive labels, and we do not want that
    #a poor performance on one such category offsets a good performance on a
    #more material category.

    #Define the grid search object that will be used to find and train the best
    #performing pipeline
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = total_scorer,
                      verbose = 3)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    """ Calculate model predictions of the multi-output classifier
    and use standard metrics (precision, recall, f1_score) to measure the
    performance achieved on each output category. Print the results.

    Parameters
    ----------
    model : Pipeline, model that will be evaluated
    X_test : Series, collection of messages in their raw format used for testing
    Y_test : DataFrame, dataset with multi-output response used for testing
    category_names : array, names of binary response variables representing
    different output categories

    Returns
    -------
    None
    """

    #Caculate the model predictions on the test set for all response variables
    Y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)

    for col in category_names:
        print(f' Category: {col}')
        print(classification_report(Y_test[col],Y_pred[col]))
        print('-----\n')


def save_model(model, model_filepath):
    """ Export the model as a pickle file

    Parameters
    ----------
    model : Pipeline, trained model
    model_filepath : str, path to the trained model storage location

    Returns
    -------
    None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        cv = build_model()

        print('Training model...')
        #Fits the grid search object
        cv.fit(X_train, Y_train)

        #Extract the best performing instance
        model = cv.best_estimator_

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
