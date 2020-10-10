# python3 train_classifier.py ./../data/DisasterResponsePipelineData.db model.pck
# database_filepath = "./../data/DisasterResponsePipelineData.db"

# import libraries
import sys
import re
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from custom_transformers import NumCharsExtractor


text_to_entity_map = pickle.load(open("text_to_entity_map.pck", "rb"))
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
lemmatizer = WordNetLemmatizer()


def getParameters():
        return {
         'vect__analyzer': ['word'],
         'vect__binary': [False],
         'vect__decode_error': ['strict'],
         'vect__encoding': ['utf-8'],
         'vect__input': ['content'],
         'vect__lowercase': [True],
         'vect__max_df': [1.0],
         'vect__max_features': [900],
         'vect__min_df': [1],
         'vect__ngram_range': ((1, 1), (1, 2)),
         'vect__preprocessor': [None],
         'vect__stop_words': [None],
         'vect__strip_accents': [None],
         'vect__token_pattern': ['[?u]\\b\\w\\w+\\b'],
         'vect__vocabulary': [None],
         'tfidf__norm': ['l2'],
         'tfidf__smooth_idf': [True],
         'tfidf__sublinear_tf': [False],
         'tfidf__use_idf': [False],
         'clf__estimator__bootstrap': [True],
         'clf__estimator__class_weight': [None],
         'clf__estimator__criterion': ['gini'],
         'clf__estimator__max_depth': [None],
         'clf__estimator__max_features': ['auto'],
         'clf__estimator__max_leaf_nodes': [None],
         'clf__estimator__min_impurity_decrease': [0.0],
         'clf__estimator__min_impurity_split': [None],
         'clf__estimator__min_samples_leaf': [1],
         'clf__estimator__min_samples_split': [2],
         'clf__estimator__min_weight_fraction_leaf': [0.0],
         'clf__estimator__n_estimators': [100],
         'clf__estimator__n_jobs': [20],
         'clf__estimator__oob_score': [False],
         'clf__estimator__random_state': [42],
         'clf__estimator__verbose': [0],
         'clf__estimator__warm_start': [False]
        }

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name="DisasterResponsePipelineData", con=engine)
    X = df[["message", "original", "genre"]]
    Y = df.iloc[:,4:]
    return X, Y, Y.columns


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #for all entities, add their type to the text
    for entity in text_to_entity_map.keys():
        if entity in text:
            text += " " + text_to_entity_map[entity]
        
    tokens = word_tokenize(text)
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(str_model = ""):
   
    if str_model == "+num_chars":
        pipeline = Pipeline([
            ('features', FeatureUnion([
    
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
    
                ('num_chars', NumCharsExtractor())
            ])),
    
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=20, random_state=42)))
        ])        
    else:
        pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=20, random_state=42)))
            ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    #predict "test" 
    Y_test_pred = model.predict(X_test)
    Y_test_pred = pd.DataFrame(data=Y_test_pred, columns=Y_test.columns)
    precision = []
    recall = []
    f1 = []
    for col in category_names:
        precision.append( precision_score(Y_test[col], Y_test_pred[col], average='weighted'))
        recall.append( recall_score(Y_test[col], Y_test_pred[col], average='weighted'))
        f1.append( f1_score(Y_test[col], Y_test_pred[col], average='weighted'))
        
    avg_precision = sum(precision) / len(category_names)
    avg_recall = sum(recall) / len(category_names)
    avg_f1 = sum(f1) / len(category_names)

    return avg_precision, avg_recall, avg_f1, precision, recall, f1


def print_evaluation(avg_precision, avg_recall, avg_f1, precision, recall, f1, category_names):
    print( " "*20 + "    precision      recall      f1-score")
    for idx, val in enumerate(category_names):
        print(val.ljust(24), end="" )
        print("%4.2f           %4.2f        %4.2f"%(precision[idx], recall[idx], f1[idx]))
        
    print("AVERAGE                 %8.6f       %8.6f    %8.6f"%(avg_precision, avg_recall, avg_f1))    
    return

def avg_f1_scorer(estimator, X, y):
    Y_pred = estimator.predict(X)
    f1 = []
    # y is a Dataframe -> use.to_numpy()
    # Y_pred is a np array
    for col in range(0, Y_pred.shape[1]):
        f1.append( f1_score(y.to_numpy()[:,col], Y_pred[:,col], average='weighted'))
        
    return sum(f1) / Y_pred.shape[1]
    

def optimize_model( model, X_test, Y_test, X_train, Y_train, category_names ):
    parameters = getParameters()
    cv = GridSearchCV(model, param_grid=parameters, cv=5, scoring = avg_f1_scorer, verbose=100, n_jobs=10)
    cv.fit(X_train, Y_train)
    
    print( "best score = %8.6f" % ( cv.best_score_ ) )
    
    return cv.best_estimator_
    


def save_model(model, model_filepath):
    pickle.dump( model, open( model_filepath, "wb" ) )
    return

def save_params(model, params_filepath):
    params = model.get_params().copy()
    pickle.dump( params, open( params_filepath, "wb" ) )
    return



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X = X["message"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        if True:
            print('Building model...')
            model = build_model()
            
            print('Optimize model...')
            model = optimize_model( model, X_test, Y_test, X_train, Y_train, category_names )
            print('Evaluating optimized model...')
            avg_precision, avg_recall, avg_f1, precision, recall, f1 = evaluate_model(model, X_test, Y_test, category_names)
            print_evaluation(avg_precision, avg_recall, avg_f1, precision, recall, f1, category_names)        
            print(model.get_params())
        
            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)
            save_params(model, "best_params.pkl")
            print('Optimized model saved!')
       
        print('Building model...')
        model_default = build_model()        
        
        print('Training model...')
        model_default.fit(X_train, Y_train)
        
        print('Evaluating model...')
        avg_precision, avg_recall, avg_f1, precision, recall, f1 = evaluate_model(model_default, X_test, Y_test, category_names)
        print_evaluation(avg_precision, avg_recall, avg_f1, precision, recall, f1, category_names)

        print('Saving default model...\n')
        save_model(model_default, "model_default.pkl")
        save_params(model_default, "model_default_params.pkl")
        print('default model saved!')        
        
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()