import re
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from pymorphy2 import MorphAnalyzer


def tokenize_dataset(dataframe: pd.DataFrame, save=False):
    """ Tokenize and lemmatize text in dataframe """
    dataframe.loc['review'] = dataframe.loc['review'].apply(lemmatize)
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop_duplicates()

    if save:
        dataframe.to_csv('tokenized_dataframe.csv', index=False)

    return dataframe


def lemmatize(review):
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.remove('не')
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    morph = MorphAnalyzer()

    review = review.lower()
    review = re.sub(patterns, ' ', review)
    tokens = []
    for token in review.split():
        if token not in russian_stopwords and len(token) > 1:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    if len(tokens) > 2:
        return ' '.join(tokens)
    return None