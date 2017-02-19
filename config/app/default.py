import os

SECRET_KEY = "oijfdoijsaoifjaosijdoija"
DEBUG = True

PICKLE_FOLDER = os.path.abspath('pickles/')
DATA_FOLDER = os.path.abspath('dataset/')
MODELS_FOLDER = os.path.abspath('ml_models/')

UNIGRAM_SENTENCES = os.path.join(DATA_FOLDER, 'unigram_sentences_all.txt')
BIGRAM_SENTENCES = os.path.join(DATA_FOLDER, 'bigram_sentences_all.txt')
RAW_REVIEWS = os.path.join(DATA_FOLDER, 'my_reviews.csv')
BIGRAM_MODEL = os.path.join(PICKLE_FOLDER, 'bigram_model_all')

SENTIMENT_VECTORISER = os.path.join(PICKLE_FOLDER, 'sentiment_vectoriser')
SENTIMENT_MODEL = os.path.join(MODELS_FOLDER, 'sentiment_model')

# lemmatized_reviews_filepath = os.path.join(DATA_FOLDER, 'lem_reviews_all.txt')

# word2vec_filepath = os.path.join(PICKLE_FOLDER, 'word2vec_model_all')
# word2vec_cluster_model_filepath = os.path.join(PICKLE_FOLDER,
#                                                'word2vec_cluster_model_all')
