import os

SECRET_KEY = "oijfdoijsaoifjaosijdoija"
DEBUG = True

PICKLE_FOLDER = os.path.abspath('../../pickles/')
DATA_FOLDER = os.path.abspath('../../dataset/')
MODELS_FOLDER = os.path.abspath('../../ml_models/')
INTERMEDIATE_FOLDER = os.path.abspath('../../dataset/intermediate/')

lemmatized_reviews_filepath = os.path.join(DATA_FOLDER,'lem_reviews_all.txt')
unigram_sentences_filepath = os.path.join(DATA_FOLDER,
                                          'unigram_sentences_all.txt')
bigram_sentences_filepath = os.path.join(DATA_FOLDER,
                                         'bigram_sentences_all.txt')
bigram_reviews_filepath = os.path.join(DATA_FOLDER,
                                        'bigram_transformed_reviews_all.txt')
bigram_model_filepath = os.path.join(PICKLE_FOLDER,'bigram_model_all')
word2vec_filepath = os.path.join(PICKLE_FOLDER,'word2vec_model_all')
word2vec_cluster_model_filepath = os.path.join(PICKLE_FOLDER,
                                               '../pickles/word2vec_cluster_model_all')
