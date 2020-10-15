# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# further description
The main technical task of the project is the classification of messages that were sent during disaster events.

These can be of one (or several) of the following categories:

1. related
1. request
1. offer
1. aid_related
1. medical_help
1. medical_products
1. search_and_rescue
1. security
1. military
1. child_alone
1. water
1. food
1. shelter
1. clothing
1. money
1. missing_people
1. refugees
1. death
1. other_aid
1. infrastructure_related
1. transport
1. buildings
1. electricity
1. tools
1. hospitals
1. shops
1. aid_centers
1. other_infrastructure
1. weather_related
1. floods
1. storm
1. fire
1. earthquake
1. cold
1. other_weather
1. direct_report


The project includes a web app where a new message can be posted and classified into the categories

## data/process_data.py

This is where the data is loaded, merged and cleaned. Afterwardsm it is written into an SQLite database from where is can be retrieved for model training.


## models/train_classifier.py

This is where the data is loaded, prepared for training and evaluation and where the model training is executed.

For optimizing the model performance, GridSearchCV is used.

The model is then saved in a pickle file from where it can be loaded for using it in e.g. the web app


## Flask Web App

This is the web app that employs the model for message classification. See the instructions at the top on how to run it.
