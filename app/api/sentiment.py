from manage import app
from app.api import api
from sklearn.model_selection import train_test_split
from app.mltrainers.dataloaders import SentimentLoader, InfLoader
from app.mltrainers.trainers import SentimentTrainer, InfTrainer
from app.mltrainers.helpers import balance_data
from flask import jsonify

raw_sent_filepath = app.config['RAW_REVIEWS']
inf_sent_filepath = app.config['INF_SENTENCES']


@api.route("/")
def congratulations():
    return "Congratulations! You have successfully launched your project"


@api.route("/sentiment/score")
def sentiment_score():
    sent_dataloader = SentimentLoader(raw_sent_filepath)
    sent_dataloader.load_data()

    data = sent_dataloader.data
    train, test = train_test_split(data, test_size=0.2)
    train = balance_data(train, 'sentiment')
    y_train = train['sentiment']
    y_test = test['sentiment']

    X_train = sent_dataloader.fit_transform(train['review'])
    X_test = sent_dataloader.transform(test['review'])

    sent_trainer = SentimentTrainer(X_train, y_train, X_test, y_test)

    sent_trainer.train()
    return jsonify(sent_trainer.score())


@api.route("/inf/score")
def inf_score():
    inf_dataloader = InfLoader(inf_sent_filepath)
    inf_dataloader.load_data()

    data = inf_dataloader.data
    train, test = train_test_split(data, test_size=0.2)
    train = balance_data(train, 'binary')
    y_train = train['binary']
    y_test = test['binary']

    X_train = inf_dataloader.fit_transform(train['review'])
    X_test = inf_dataloader.transform(test['review'])

    inf_trainer = InfTrainer(X_train, y_train, X_test, y_test)

    inf_trainer.train()
    return jsonify(inf_trainer.score())
