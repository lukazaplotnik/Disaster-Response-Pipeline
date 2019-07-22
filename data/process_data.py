import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load messages and corresponding categorizations, and merge them into
    a dataframe based on the message ID.

    Parameters
    ----------
    messages_filepath: str, path to the csv file with messages
    categories_filepath: str, path to the csv file with categorizations

    Returns
    -------
    df: DataFrame, merged dataset
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on = ['id'])

    return df

def clean_data(df):
    """ Clean the dataframe by splitting the categories column into separate,
    clearly named columns, converting values to binary, and dropping duplicates.

    Parameters
    ----------
    df: DataFrame, uncleaned dataframe

    Returns
    -------
    df: DataFrame, cleaned dataframe
    """

    #Split the categories column into separate columns
    categories = df['categories'].str.split(";", expand=True)

    #Extract a list of new column names for categories dataset
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #Extract only the numeric values from the strings in categories columns,
    #e.g. 1 from 'related-1' and 0 from 'request-0'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    #Ensure there are only binary 0/1 values in the dataset
    categories[categories>1] = 1

    #Remove columns that have all 0 values
    categories = categories.drop(categories.columns[categories.sum() == 0],
                                 axis=1)

    #Drop the original categories column from the uncleaned dataframe
    df = df.drop('categories', axis=1)

    #Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    #Drop duplicate rows based on the ID column.

    #Note:There are some instances of duplicate IDs that do not contain the same
    #content in the 'categories' column. Ideally, one would manually select
    #which row to keep and which to delete based on the content in a given
    #message. However, the aim is to develop an automated ETL approach,
    #therefore we rely on a simpler approach and directly use the
    #drop_duplicates method, keeping only the first instance of each duplicate
    #ID. The simplification does not lead to significant loss of information,
    #as the number of problematic rows is low.
    df = df.drop_duplicates('id')

    return df


def save_data(df, database_filename):
    """ Store the clean dataframe into a SQLite database in the specified
    database file path

    Parameters
    ----------
    df: DataFrame, cleaned dataframe
    database_filename: str, specified database file path (relative)

    Returns
    -------
    None
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
