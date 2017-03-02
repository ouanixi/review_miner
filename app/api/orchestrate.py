from sklearn.model_selection import train_test_split

from app.mltrainers.dataloaders import SentimentLoader, InfLoader, IntentLoader
from app.mltrainers.helpers import balance_data
from app.mltrainers.trainers import SentimentTrainer, InfTrainer, IntentTrainer
from app.utils.nlp_utils import make_sentences


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
        review_dict['sentiment'] = analyse_sentiment(review_dict.get('review'))
        review_dict['sentences'] = analyse_inf(review_dict['review'])
        review_dict['intent'] = analyse_intent(review_dict['sentences'])
    return reviews_list


def analyse_sentiment(review):
    trainer = SentimentTrainer()
    loader = SentimentLoader()
    vector = loader.transform([review])
    sentiment = trainer.predict(vector)
    return str(sentiment[0])


def analyse_inf(review):
    sentences = list(make_sentences(review))
    trainer = InfTrainer()
    loader = InfLoader()
    matrix = loader.transform(sentences)
    inf = trainer.predict(matrix)
    sent_dict = dict(zip(sentences, [str(i) for i in inf]))
    return sent_dict


def analyse_intent(review_sent_dict):
    intent_dict = {}
    sentences = [sent for sent in review_sent_dict.keys() if review_sent_dict[sent] == "1"]
    if sentences:
        trainer = IntentTrainer()
        loader = IntentLoader()
        matrix = loader.transform(sentences)
        intent = trainer.predict(matrix)
        intent_dict = dict(zip(sentences, [str(i) for i in intent]))
    return intent_dict
