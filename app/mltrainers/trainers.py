import os.path

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from app.mltrainers.exceptions import NoPickleAvailable
from manage import app

sentiment_classifier_filepath = app.config['SENTIMENT_MODEL']
inf_classifier_filepath = app.config['INF_MODEL']


class Trainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @property
    def model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)

    def score(self):
        predicted = self.model.predict(self.X_test)
        precision = metrics.precision_score(self.y_test, predicted)
        recall = metrics.recall_score(self.y_test, predicted)
        accuracy = metrics.accuracy_score(self.y_test, predicted)
        f1_score = metrics.f1_score(self.y_test, predicted)
        return {'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1_score}


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
