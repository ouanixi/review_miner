import os.path

import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.mltrainers.exceptions import NoPickleAvailable
from app.utils import nlp_utils as cm
from manage import app
import numpy as np
import spacy

sentiment_vectoriser_filepath = app.config['SENTIMENT_VECTORISER']
word2vec_filepath = app.config['WORD2VEC']
bigram_sentences_filepath = app.config['BIGRAM_SENTENCES']
word2vec_cluster_model_filepath = app.config['WORD2VEC_KMEANS_MODEL']


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
        vectoriser = TfidfVectorizer(decode_error="ignore",
                                     analyzer='word', lowercase=True)
        vectorised = vectoriser.fit_transform(new_data)
        joblib.dump(vectoriser, sentiment_vectoriser_filepath)
        return vectorised

    def transform(self, new_data):
        return self.vectoriser.transform(new_data)

    def load_data(self):
        data = pd.read_csv(self.filepath)
        data = data[data['score'] != 3]
        data['review'] = data['review'].apply(cm.remove_punctuation)
        data['sentiment'] = data['score'].apply(
            lambda score: +1 if score > 3 else -1)
        data = data[['review', 'sentiment']].dropna()
        self.data = data


class InfLoader(DataLoader):
    def __init__(self, filepath):
        super().__init__(self, filepath)
        self.word2vec = Word2vecLoader()

    @property
    def vectoriser(self):
        if os.path.exists(word2vec_filepath):
            kmeans_clustering = joblib.load(word2vec_cluster_model_filepath)
            return kmeans_clustering
        return self._make_clusers()

    def load_data(self):
        self.data = pd.read_csv(self.filepath).dropna()

    def fit_transform(self, new_sentences):
        """
        Functionality doesn't apply for this loader. So returning results of transform
        """
        return self.transform(new_sentences)

    def transform(self, new_sentences):
        """
        transforms a list of sentences into a matrix of centroid vectors.
        Note that for best results, the sentences should be lemmatized before
        calling this method.

        :param new_data: list of lemmatized_sentences
        :return: feature vectors
        """

        # Need to keep reference of the model since calling it using .model always reloads the
        # same one from file.
        wor2vec = self.word2vec.model
        number_of_examples = len(new_sentences)
        wor2vec.build_vocab(self._prepare_for_train(new_sentences), update=True)
        wor2vec.train(self._prepare_for_train(new_sentences), total_example=number_of_examples)
        num_clusters = len(np.unique(self.vectoriser.labels_))
        # Pre-allocate an array for the training set bags of centroids (for speed)
        features = np.zeros((number_of_examples, num_clusters), \
                                   dtype="float32")
        counter = 0
        for review in self._prepare_for_train(new_sentences):
            features[counter] = self._build_word2vec_feature(review)
            counter += 1
        return features

    def _make_clusers(self):
        kmeans_clustering = KMeans(n_clusters=220, n_jobs=-2).fit(self.word2vec.model.wv.syn0)
        joblib.dump(kmeans_clustering, word2vec_cluster_model_filepath)
        return kmeans_clustering

    def _build_word2vec_feature(self, bigram_sent):
        labels = self.vectoriser.labels_
        labels_unique = np.unique(labels)
        num_clusters = len(labels_unique)

        features = np.zeros(num_clusters)
        idx_count = {}

        for word in bigram_sent:
            try:
                cluster = self.vectoriser.predict(self.word2vec.model[word].reshape(1, -1))[0]
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
            parsed_review = cm.nlp(review)
            unigram_review = [token for token in parsed_review
                              if not cm.punct_space(token)]
            # apply the first-order and second-order phrase models
            bigram_review = cm.get_bigram_model()[unigram_review]
            yield bigram_review


class Word2vecLoader():
    def __init__(self):
        self.data = cm.get_bigram_sentences()

    @property
    def model(self):
        if os.path.exists(word2vec_filepath):
            app2vec = Word2Vec.load(word2vec_filepath)
            app2vec.init_sims(replace=False)
            return app2vec
        raise NoPickleAvailable(
            "No word2vec model trained! Train one first")

    def initialise(self):
        app2vec = Word2Vec(size=300, window=5, min_count=10, sg=1, workers=7, iter=12)
        app2vec.build_vocab(self.data)
        app2vec.train(self.data)
        app2vec.save(word2vec_filepath)
        return app2vec

    def add(self, sentences):
        """
        Assumes model pre-trained already and adds new sentences to the vocab
        :param sentences: Generator object of type LineSentence or similar generators
        :return: newly trained word2vec model
        """
        app2vec = self.model
        app2vec.build_vocab(sentences, update=True)
        app2vec.train(sentences)
