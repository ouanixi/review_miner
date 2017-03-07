"""Loader classes for the different classification tasks used in this project."""
import logging
import os.path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.mltrainers.exceptions import NoPickleAvailable
from app.utils import generators as gen
from app.utils.generators import Singleton
from app.utils import nlp_utils as cm
from config.app.default import (SENTIMENT_VECTORISER, WORD2VEC,
                                BIGRAM_SENTENCES, KMEANS_MODEL,
                                RAW_REVIEWS, INF_SENTENCES,
                                INTENT_SENTENCES)

sentiment_vectoriser_filepath = SENTIMENT_VECTORISER
word2vec_filepath = WORD2VEC
bigram_sentences_filepath = BIGRAM_SENTENCES
cluster_model_filepath = KMEANS_MODEL
raw_reviews_filepath = RAW_REVIEWS
inf_sent_filepath = INF_SENTENCES
intent_sent_filepath = INTENT_SENTENCES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class DataLoader(metaclass=Singleton):
    """Interface class exposing data preparation API."""

    def __init__(self):
        """Constructor."""
        self.filepath = None
        self.data = None
        self.vectoriser = None

    @property
    def get_vector(self):
        """Stub method that loads class vectoriser."""
        raise NotImplementedError

    def load_data(self):
        """Load appropriate data sources for classification task."""
        raise NotImplementedError

    def fit_transform(self, data):
        """Make initial vectorisation task from empty vectoriser."""
        raise NotImplementedError

    def transform(self, raw):
        """Vectorise data based on existing vectoriser."""
        raise NotImplementedError


class SentimentLoader(DataLoader):
    """Load and prepare sentiment analysis data to be used in training."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.filepath = raw_reviews_filepath

    @property
    def get_vector(self):
        """Load TFIDF vectoriser.

        @raise NoPickleAvailable: raised when no vectoriser found default path.
        """
        if os.path.exists(sentiment_vectoriser_filepath):
            return joblib.load(sentiment_vectoriser_filepath)
        raise NoPickleAvailable(
            "No vectoriser available! Make sure you ran fit_transorm first")

    def fit_transform(self, new_data):
        """Make initial TFIDF vectorisation task from empty vectoriser."""
        vectoriser = TfidfVectorizer(decode_error="ignore",
                                     analyzer='word', lowercase=True)
        vectorised = vectoriser.fit_transform(new_data)
        joblib.dump(vectoriser, sentiment_vectoriser_filepath)
        return vectorised

    def transform(self, new_data):
        """Vectorise data based on existing TFIDF vectoriser."""
        return self.vectoriser.transform(new_data)

    def load_data(self):
        """Load appropriate data sources for sentiment classification task."""
        data = pd.read_csv(self.filepath)
        data = data[data['score'] != 3]
        data['review'] = data['review'].apply(cm.remove_punctuation)
        data['review'] = data['review'].str.strip()
        data['class'] = data['score'].apply(
            lambda score: +1 if score > 3 else -1)
        data = data[['review', 'class']].dropna()
        self.data = data


class InfLoader(DataLoader):
    """Load and prepare informative vs non-informative data to be used in training."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.filepath = inf_sent_filepath
        self.word2vec = None
        self.bigram_model = None

    @property
    def get_vector(self):
        """Load TFIDF vectoriser.

        @raise NoPickleAvailable: raised when no vectoriser found default path.
        """
        if os.path.exists(cluster_model_filepath):
            clstr = joblib.load(cluster_model_filepath)
            return clstr
        return self._make_clusers()

    def load_data(self):
        """Load appropriate data sources for inf vs non-inf classification task."""
        data = pd.read_csv(self.filepath).dropna()
        data['class'] = data['class'].apply(lambda s: -1 if s ==
                                            'non-informative' else 1)
        self.data = data

    def fit_transform(self, new_sentences):
        """
        Functionality doesn't apply for this loader.

        Returning results of transform
        """
        return self.transform(new_sentences)

    def transform(self, new_sentences):
        """Transform a list of sentences into a matrix of centroid vectors.

        It uses a pretrained Word2Vec model clustered into 220 clusters
        Note that for best results, the sentences should be lemmatized
        before calling this method.

        :param new_sentences: list of raw sentences
        :return: feature vectors of size 220
        """
        # Need to keep reference of the model since calling it using
        # .model always reloads the same one from file.
        word2vec = self.word2vec.model
        number_of_examples = len(new_sentences)
        logger.info("Adding vocab to word2vec model!")
        word2vec.build_vocab(self._prepare_for_train(new_sentences),
                             update=True)
        word2vec.train(self._prepare_for_train(new_sentences),
                       total_examples=number_of_examples)
        clstr = self.vectoriser
        num_clusters = len(np.unique(clstr.labels_))
        # Pre-allocate an array for the training set bags of
        # centroids (for speed)
        logger.info("Generating bag of centroids matrix")
        features = np.zeros((number_of_examples, num_clusters),
                            dtype="float32")
        counter = 0
        for bigram_sent in self._prepare_for_train(new_sentences):
            features[counter] = self._build_word2vec_feature(bigram_sent,
                                                             word2vec,
                                                             num_clusters,
                                                             clstr)
            counter += 1
        logger.info("Bag of centroids matrix generated")
        return features

    def _make_clusers(self):
        logger.info("Clustering task started!")
        clstr = KMeans(n_clusters=220, n_jobs=-2).fit(self.word2vec.wv.syn0)
        joblib.dump(clstr, cluster_model_filepath)
        logger.info("Clustering task completed!")
        return clstr

    def _build_word2vec_feature(self, bigram_sent, word2vec,
                                num_clusters, clstr):
        features = np.zeros(num_clusters)
        idx_count = {}

        for word in bigram_sent:
            try:
                cluster = clstr.predict(word2vec[word].reshape(1, -1))[0]
                if idx_count.get(cluster):
                    idx_count[cluster] += 1
                else:
                    idx_count[cluster] = 1
            except KeyError:
                pass
        for key in idx_count:
            features[key] = idx_count[key]
        return features

    def _prepare_for_train(self, new_sentences):
        for review in new_sentences:
            parsed_review = gen.NLP().nlp(review)
            unigram_review = [token.lemma_ for token in parsed_review
                              if not cm.punct_space(token)]
            # apply the first-order and second-order phrase models
            bigram_review = self.bigram_model[unigram_review]
            yield bigram_review


class IntentLoader(InfLoader):
    """Load and prepare User Intention data to be used in training.

    Inherits from InfLoader to share transform and fit_transform methods.
    """

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.filepath = intent_sent_filepath

    def load_data(self):
        """Load appropriate data sources for intent classification task."""
        data = pd.read_csv(self.filepath).dropna()
        data['review'] = data['review'].str.strip()
        data['NLP_classification'] = data['NLP_classification'].str.strip()
        data['class'] = data['NLP_classification'].apply(self._make_class)
        self.data = data

    def _make_class(self, classification):
        if classification == u'problem discovery':
            return 0
        if classification == u'feature request':
            return 1
        if classification == u'information seeking':
            return 2
        else:  # information giving
            return 3


class Word2vecLoader(metaclass=Singleton):
    """Responsible for training or loading Word Embeddings."""

    def __init__(self):
        """Constructor."""
        self.data = None
        self.model = None

    @property
    def get_model(self):
        """Load Word2Vec model.

        @raise NoPickleAvailable: raised when no vectoriser found in default path.
        """
        if os.path.exists(word2vec_filepath):
            app2vec = Word2Vec.load(word2vec_filepath)
            app2vec.init_sims(replace=False)
            return app2vec
        raise NoPickleAvailable(
            "No word2vec model trained! Train one first")

    def load_data(self):
        """Load bigram sentences."""
        self.data = cm.get_bigram_sentences()

    def initialise(self):
        """Train Word2Vec model from scratch."""
        app2vec = Word2Vec(size=300, window=5, min_count=10, sg=1,
                           workers=7, iter=12)
        app2vec.build_vocab(self.data)
        app2vec.train(self.data)
        app2vec.save(word2vec_filepath)
        return app2vec

    def add(self, sentences):
        """Assume model pre-trained already and adds new sentences to the vocab.

        :param sentences: Generator object of type LineSentence or
        similar generators
        :return: newly trained word2vec model
        """
        app2vec = self.model
        app2vec.build_vocab(sentences, update=True)
        app2vec.train(sentences)
