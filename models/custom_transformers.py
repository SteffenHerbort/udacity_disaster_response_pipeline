import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag


class NumCharsExtractor(BaseEstimator, TransformerMixin):
    '''
	Transformer to be used in a pipieline / feature union.
    Extracts the length of each document in an input dataset (X)

			Parameters:
					None

	'''
    def fit(self, x, y=None):
        '''
    	fit-method required for the class to function as a transformer/estimator
        
        can be called, but does nothing
        
                Returns:
                    self
            
    
    	'''        
        return self

    def transform(self, X):
        '''
    	transforms the X (input) data
    
    			Parameters:
    					X (pandas DataFrame): input data
    
    			Returns:
    					X_num_chars (pandas DataFrame): Nx1 dataframe where 
                                                        each element specifies 
                                                        the number of 
                                                        characters in 
                                                        the respective text
    	'''
        # extract number of characters from a document ("text")
        X_num_chars = np.ones(X.shape, dtype=bool)
        for idx, elem in enumerate(X):
            X_num_chars[idx] = len( elem )

        return pd.DataFrame(X_num_chars)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
	Transformer to be used in a pipieline / feature union.
    Extracts, if a document in an input dataset (X) starts with a verb

			Parameters:
					None

	'''
    
    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize( text )
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag( word_tokenize( sentence ) )

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

            return False

    def fit(self, x, y=None):
        '''
    	fit-method required for the class to function as a transformer/estimator
        
        can be called, but does nothing
        
                Returns:
                    self
                
    
    	'''            
        return self

    def transform(self, X):
        '''
    	transforms the X (input) data
    
    			Parameters:
    					X (pandas DataFrame): input data
    
    			Returns:
    					X_tagged (pandas DataFrame): Nx1 dataframe where 
                                                     each element specifies if
                                                     the respective text starts
                                                     with a verb
    	'''        
        # apply starting_verb function to all values in X
        X_tagged = np.ones(X.shape, dtype=bool)
        for idx, elem in enumerate(X):
            X_tagged[idx] = self.starting_verb( elem )

        return pd.DataFrame(X_tagged)