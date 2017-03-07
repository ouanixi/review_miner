"""Exposes classes relevant to ML training."""
import os.path

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from app.mltrainers.exceptions import NoPickleAvailable
from manage import app

sentiment_classifier_filepath = app.config['SENTIMENT_MODEL']
inf_classifier_filepath = app.config['INF_MODEL']
intent_classifier_filepath = app.config['INTENT_MODEL']


class Trainer:
    """Abstract class."""

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """Constructor."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @property
    def model(self):
        """Abstract method."""
        raise NotImplementedError

    def train(self):
        """Abstract method."""
        raise NotImplementedError

    def predict(self, X):
        """Concrete method."""
        return self.model.predict(X)

    def score(self):
        """Concrete method."""
        predicted = self.model.predict(self.X_test)
        precision = metrics.precision_score(self.y_test, predicted)
        recall = metrics.recall_score(self.y_test, predicted)
        accuracy = metrics.accuracy_score(self.y_test, predicted)
        f1_score = metrics.f1_score(self.y_test, predicted)
        score = {'precision': precision,
                 'recall': recall,
                 'accuracy': accuracy,
                 'f1_score': f1_score}
        return score


class SentimentTrainer(Trainer):
    @property
    def model(self):
        if os.path.exists(sentiment_classifier_filepath):
            return joblib.load(sentiment_classifier_filepath)
        raise NoPickleAvailable("No classifier avaiable")

    def train(self):
        clf = LogisticRegression(n_jobs=-2).fit(self.X_train, self.y_train)
        joblib.dump(clf, sentiment_classifier_filepath)


class InfTrainer(Trainer):
    @property
    def model(self):
        if os.path.exists(inf_classifier_filepath):
            return joblib.load(inf_classifier_filepath)
        raise NoPickleAvailable("No classifier avaiable")

    def train(self):
        clf = LogisticRegression(n_jobs=-2).fit(self.X_train, self.y_train)
        joblib.dump(clf, inf_classifier_filepath)


class IntentTrainer(Trainer):
    @property
    def model(self):
        if os.path.exists(intent_classifier_filepath):
            return joblib.load(intent_classifier_filepath)
        raise NoPickleAvailable("No classifier avaiable")

    def train(self):
        clf = GradientBoostingClassifier(n_estimators=100,
                                         learning_rate=1.0,
                                         max_depth=2,
                                         random_state=0)
        clf.fit(self.X_train, self.y_train)
        joblib.dump(clf, intent_classifier_filepath)

    def score(self):
        predicted = self.model.predict(self.X_test)
        precision = metrics.precision_score(self.y_test, predicted, average='weighted')
        recall = metrics.recall_score(self.y_test, predicted, average='weighted')
        accuracy = metrics.accuracy_score(self.y_test, predicted)
        f1_score = metrics.f1_score(self.y_test, predicted, average='weighted')
        score = {'precision': precision,
                 'recall': recall,
                 'accuracy': accuracy,
                 'f1_score': f1_score}
        return score
