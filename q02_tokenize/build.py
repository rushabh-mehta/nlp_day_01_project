# %load q02_tokenize/build.py
# Default imports

from nltk.tokenize import TreebankWordTokenizer

import pandas as pd

from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
path = 'data/20news-bydate-train/'
# Write your solution here:
def q02_tokenize(path):
    file,X_train,X_test,y_train,y_test = q01_load_data(path)
    tree_bank_tokenizer = TreebankWordTokenizer()
    X_train = pd.Series(X_train)
    X_test = pd.Series(X_test)
    X_train = X_train.apply(lambda x : x.lower())
    X_test = X_test.apply(lambda x : x.lower())
    i=0
    for row in X_train:
        X_train.iloc[i]=tree_bank_tokenizer.tokenize(str(row))
        i+=1
    return X_train


