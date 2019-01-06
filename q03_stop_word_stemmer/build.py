# %load q03_stop_word_stemmer/build.py
# Default imports
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data
from greyatomlib.nlp_day_01_project.q02_tokenize.build import q02_tokenize
from nltk.corpus import stopwords
import pandas as pd
stop = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer


path = 'data/20news-bydate-train/'
# Your solution here:
def q03_stop_word_stemmer(path):
    X_train = q02_tokenize(path)
    stop_words=set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    
    i=0
    for row in X_train:
        sentence=[]
        for word in row:
            if word not in stop_words:
                porter_stemmer.stem(word)
                sentence.append(word)
        X_train.iloc[i]=sentence
        i+=1
    return list(X_train)


