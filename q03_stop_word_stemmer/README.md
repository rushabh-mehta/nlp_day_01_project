# Removing stop words and performing stemming

This assignment comprises of implementation of one of the removal of stop words and finally applying 
the preprocessing step namely `stemming`.


## Write a function `q03_stop_word_stemmer` that 
- Makes use of input `q01_load_data`.
- Converts the training data `X_train` to series.
- Removes the stop words from the training data .
- Iterates over the rows obtained from previous step (i.e text processed after applying stop words) and performs stemming using the PorterStemmer. 



### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | String | compulsory |  | Path of data folder |




### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable | list | stemmed text belonging to each category of data |
