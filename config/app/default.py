import os

SECRET_KEY = "oijfdoijsaoifjaosijdoija"
DEBUG = True
PICKLE_FOLDER = os.path.abspath('../../pickles/')
DATA_FOLDER = os.path.abspath('../../dataset/')
MODELS_FOLDER = os.path.abspath('../../ml_models/')
INTERMEDIATE_FOLDER = os.path.abspath('../../dataset/intermediate/')

unigram_sentences_filepath = os.path.join(DATA_FOLDER,'unigram_sentences_all.txt')
bigram_sentences_filepath = os.path.join(DATA_FOLDER, 'bigram_sentences_all.txt')
bigram_bow_filepath = os.path.join(DATA_FOLDER,'bigram_bow_corpus_all.mm')
trigram_sentences_filepath = os.path.join(DATA_FOLDER,'trigram_sentences_all.txt')
trigram_reviews_filepath = os.path.join(DATA_FOLDER,'trigram_transformed_reviews_all.txt')
lemmatized_reviews_filepath = os.path.join(DATA_FOLDER,'lem_reviews_all')

lda_model_filepath = os.path.join(PICKLE_FOLDER,'lda_model_all')
bigram_model_filepath = os.path.join(PICKLE_FOLDER,'bigram_model_all')
trigram_model_filepath = os.path.join(PICKLE_FOLDER,'trigram_model_all')

bigram_dictionary_filepath = os.path.join(PICKLE_FOLDER,'bigram_dict_all.dict')
trigram_dictionary_filepath = os.path.join(PICKLE_FOLDER,'trigram_dict_all.dict')

trigram_bow_filepath = os.path.join(DATA_FOLDER,'trigram_bow_corpus_all.mm')

word2vec_filepath = os.path.join(PICKLE_FOLDER,'word2vec_model_all')
