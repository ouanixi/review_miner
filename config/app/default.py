import os

SECRET_KEY = "oijfdoijsaoifjaosijdoija"
DEBUG = True

PICKLE_FOLDER = os.path.abspath('pickles/')
DATA_FOLDER = os.path.abspath('dataset/')
MODELS_FOLDER = os.path.abspath('ml_models/')
SCORES_FOLDER = os.path.abspath('scores/')

# Reviews and Sentences
UNIGRAM_SENTENCES = os.path.join(DATA_FOLDER, 'unigram_sentences_all.txt')
BIGRAM_SENTENCES = os.path.join(DATA_FOLDER, 'bigram_sentences_all.txt')
POSTAGGED_SENTENCES = os.path.join(DATA_FOLDER, 'postagged_sentences_all.txt')
RAW_REVIEWS = os.path.join(DATA_FOLDER, 'my_reviews.csv')
INF_SENTENCES = os.path.join(DATA_FOLDER, 'inf_reviews.csv')
INTENT_SENTENCES = os.path.join(DATA_FOLDER, 'panicella_ds.csv')

# Classifiers:
SENTIMENT_MODEL = os.path.join(MODELS_FOLDER, 'sentiment_model')
INF_MODEL = os.path.join(MODELS_FOLDER, 'inf_model')
INTENT_MODEL = os.path.join(MODELS_FOLDER, 'intent_model')

# Other pickles
BIGRAM_MODEL = os.path.join(PICKLE_FOLDER, 'bigram_model_all')
SENTIMENT_VECTORISER = os.path.join(PICKLE_FOLDER, 'sentiment_vectoriser')
WORD2VEC = os.path.join(PICKLE_FOLDER, 'word2vec_model_all')
KMEANS_MODEL = os.path.join(PICKLE_FOLDER, 'kmeans_model_all')

# scores
SENTIMENT_SCORE = os.path.join(SCORES_FOLDER, 'sentiment_score')
INF_SCORE = os.path.join(SCORES_FOLDER, 'inf_score')
INTENT_SCORE = os.path.join(SCORES_FOLDER, 'intent_score')

