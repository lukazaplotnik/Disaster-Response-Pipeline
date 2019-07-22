# Disaster Response Pipeline Project

### Project Summary:
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

By using real disaster text messages that were sent during disaster events we train a machine learning pipeline that is able to categorize disaster events from a given text, and can ultimately be used to notify appropriate disaster relief agencies. For this purpose, also a Flask web app has been implemented that would allow an emergency worker to input a new message and obtain classification results for several disaster event categories.

### File Descriptions:

data/disaster_messages.csv: input file with disaster response messages
data/disaster_categories.csv: input file with the categorization of the input
disaster response messages
data/process_data.py : implements a data cleaning pipeline that:
- loads the messages and categories datasets
- merges the two datasets
- cleans the data
- stores it in a SQLite database

models/train_classifier.py: implements a machine learning pipeline that:
- loads data from the SQLite database
- splits the dataset into training and test sets
- builds a pipeline that combines text transformation (tf-idf approach) and
a multi-output classifier (one RandomForestClassifier per output category)
- trains and tunes a pipeline using GridSearchCV
- outputs results on the test set
- exports the final model as a pickle file

app/templates/master.html: displays visuals and receives user input text
for classification
app/templates/go.html: handles user query and displays model results
app/run.py: implements functions that run a Flask web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
