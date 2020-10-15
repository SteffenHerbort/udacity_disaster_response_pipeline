# python3 ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponsePipelineData.db

import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
	Load messages and categories from files

			Parameters:
					messages_filepath (string): filename of the csv-file 
                                                containing the messages
                    categories_filepath (string): filename of the csv-file 
                                                  containing the categories

			Returns:
					df (Pandas DataFrame)
	'''       
    
    #-- load from file
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #-- merge based on "id"
    df = messages.merge(categories, on="id")

    return df
    
    


def clean_data(df):
    
    '''
	Clean text data in a Dataframe. Mainly, the text labels are transformed
    into columns with the label name and entries with the label value.

			Parameters:
					df (Pandas DataFrame): dataframe containing the raw 
                                           messages and categories data

			Returns:
					df (Pandas DataFrame)
	'''       
    
    
    #-- now, take care of the column names
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # set these as the column names
    categories.columns = category_colnames

    #-- Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    
    #-- Replace categories column in df with new category columns.
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    #-- drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
	Save dataframe in an SQL database

			Parameters:
					df (Pandas DataFrame): dataframe containing the 
                                           messages and categories data
					database_filename (string): filename of the target SQL 
                                                database
			Returns:
					void
	'''       
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponsePipelineData', engine, index=False, if_exists = "replace")


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