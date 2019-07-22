import json
import plotly
import pandas as pd
import seaborn as sns

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    #Original visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Extracted data for custom visuals - relative number of positives per category
    categories_positives = (df.iloc[:,4:].sum()/df.shape[0]).sort_values(ascending = False).round(4)
    categories_names = list(categories_positives.index)
   
    #Extracted data for custom visuals - histogram of message lengths (without outliers)
    lengths = df['message'].apply(lambda x: len(x))
    q1 = lengths.quantile(0.25)
    q3 = lengths.quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 1.5*IQR
    upper_bound = q3 + 1.5*IQR

    lengths_plot = lengths[(lengths>lower_bound)&(lengths<upper_bound)]
    
    # create visuals
    graphs = [
        #Distribution of genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        #Distribution of lengths
        {
            'data': [
                Histogram(
                    x=lengths_plot
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Length"
                }
            }
        },
        
        #Relative number of positives per category
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_positives,
                    orientation = "v"
                )
            ],

            'layout': {
                'title': 'Relative Number of Positives in Each Category',
                'yaxis': {
                    'title': "Relative Number"
                },
                'xaxis': {
                    'title': "Category"
                },
                'margin':{
                    'b': 140   
                }             
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()