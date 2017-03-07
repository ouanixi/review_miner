import os
from flask import Flask
from app.mltrainers.dataloaders import SentimentLoader, InfLoader, IntentLoader, Word2vecLoader
from app.utils.nlp_utils import get_bigram_model

sent_loader = SentimentLoader()
inf_loader = InfLoader()
intent_loader = IntentLoader()
word2vec = Word2vecLoader()

def bootstrap():
    instance_path = os.path.abspath(os.path.join(__file__, os.pardir,
                                                 "config"))
    app = Flask(__name__, instance_path=instance_path,
                instance_relative_config=True)

    load_app_config(app)
    sent_loader.vectoriser = sent_loader.get_vector
    inf_loader.vectoriser = inf_loader.get_vector
    intent_loader.vectoriser = intent_loader.get_vector
    word2vec.model = word2vec.get_model
    inf_loader.word2vec = word2vec
    intent_loader.word2vec = word2vec
    inf_loader.bigram_model = get_bigram_model()
    intent_loader.bigram_model = get_bigram_model()

    from app.api import api as api_blueprint

    app.register_blueprint(api_blueprint)

    return app


def load_app_config(app):
    app.config.from_object("config.app.default")
    app.config.from_envvar("APP_CONFIG", silent=True)
    app.config["VERSION"] = os.environ.get("VERSION", "local")
    app.config["HOSTNAME"] = os.environ.get("HOSTNAME")
