import logging
import os.path

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from app.mltrainers.exceptions import NoPickleAvailable
from app.utils import corpmaker as cm
from manage import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

sentiment_vectoriser_filepath = app.config['SENTIMENT_VECTORISER']


class DataLoader:
    """ Interface exposing data preparation API """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    @property
    def vectoriser(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def fit_transform(self, data):
        raise NotImplementedError

    def transform(self, raw):
        raise NotImplementedError


class SentimentLoader(DataLoader):
    @property
    def vectoriser(self):
        if os.path.exists(sentiment_vectoriser_filepath):
            return joblib.load(sentiment_vectoriser_filepath)
        raise NoPickleAvailable(
            "No vectoriser available! Make sure you ran fit_transorm first")

    def fit_transform(self, new_data):
        logger.info("Fitting vectorizer")
        vectoriser = TfidfVectorizer(decode_error="ignore",
                                     analyzer='word', lowercase=True)
        vectorised = vectoriser.fit_transform(new_data)
        joblib.dump(vectoriser, sentiment_vectoriser_filepath)
        return vectorised

    def transform(self, new_data):
        return self.vectoriser.transform(new_data)

    def load_data(self):
        logger.info("Loading data")

        data = pd.read_csv(self.filepath)
        data = data[data['score'] != 3]
        data['review'] = data['review'].apply(cm.remove_punctuation)
        data['sentiment'] = data['score'].apply(
            lambda score: +1 if score > 3 else -1)
        data = data[['review', 'sentiment']].dropna()
        self.data = data

        logger.info("Data loaded")

    def _get_raw_train_test(self):
        raw_data = self.data
        return train_test_split(raw_data, test_size=0.2, random_state=1)
