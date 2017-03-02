import logging

from sklearn.model_selection import train_test_split

from app.bootstrap import sent_loader, inf_loader, intent_loader
from app.mltrainers.dataloaders import SentimentLoader, InfLoader, IntentLoader
from app.mltrainers.helpers import balance_data
from app.mltrainers.trainers import SentimentTrainer, InfTrainer, IntentTrainer
from app.utils.nlp_utils import make_sentences

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class ClassificationNotSupported(Exception):
    pass


def prepare_trainer(classification=None):
    if classification == 'sentiment':
        dataloader = SentimentLoader()
        trainer = SentimentTrainer()
    elif classification == 'inf':
        dataloader = InfLoader()
        trainer = InfTrainer()
    elif classification == 'intent':
        dataloader = IntentLoader()
        trainer = IntentTrainer()
    else:
        raise ClassificationNotSupported

    dataloader.load_data()

    data = dataloader.data
    train, test = train_test_split(data, test_size=0.2, random_state=2)
    if classification != 'intent':
        train = balance_data(train, 'class')
    y_train = train['class']
    y_test = test['class']

    X_train = dataloader.fit_transform(train['review'])
    X_test = dataloader.transform(test['review'])

    trainer.X_train = X_train
    trainer.y_train = y_train
    trainer.X_test = X_test
    trainer.y_test = y_test

    return trainer


def summarise_reviews(reviews_list):
    for review_dict in reviews_list:
        logger.info("Analysing sentiments for reviewid {}".format(review_dict['id']))
        review_dict['sentiment'] = analyse_sentiment(review_dict.get('review'))
        logger.info("Analysing informative vs non informative sentences for reviewid {}".format(review_dict['id']))
        review_dict['sentences'] = analyse_inf(review_dict['review'])
        logger.info("Analysing user intents for reviewid {}".format(review_dict['id']))
        review_dict['intent'] = analyse_intent(review_dict['sentences'])
    return reviews_list


def analyse_sentiment(review):
    trainer = SentimentTrainer()
    vector = sent_loader.transform([review])
    sentiment = trainer.predict(vector)
    return str(sentiment[0])


def analyse_inf(review):
    sentences = list(make_sentences(review))
    trainer = InfTrainer()
    matrix = inf_loader.transform(sentences)
    inf = trainer.predict(matrix)
    sent_dict = dict(zip(sentences, [str(i) for i in inf]))
    return sent_dict


def analyse_intent(review_sent_dict):
    intent_dict = {}
    sentences = [sent for sent in review_sent_dict.keys() if review_sent_dict[sent] == "1"]
    if sentences:
        trainer = IntentTrainer()
        matrix = intent_loader.transform(sentences)
        intent = trainer.predict(matrix)
        intent_dict = dict(zip(sentences, [str(i) for i in intent]))
    return intent_dict
