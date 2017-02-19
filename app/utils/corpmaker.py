import codecs
import logging
import os.path

import spacy
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from app.utils import generators as gen
from manage import app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

unigram_sentences_filepath = app.config['UNIGRAM_SENTENCES']
bigram_sentences_filepath = app.config['BIGRAM_SENTENCES']
bigram_model_filepath = app.config['BIGRAM_MODEL']
RAW_REVIEWS = app.config['RAW_REVIEWS']


def make_corpus():
    """ Takes in a path for the raw corpus and creates
        two news corpora:
        1- unigram sentences
        2- bigram sentences
    """
    # logger.info("making unigram sentences")
    # save_unigram_sentences(RAW_REVIEWS)
    logger.info("making bigram sentences")
    save_bigram_sentences()


def save_unigram_sentences(filename):
    """ Saves unigram sentences to file."""
    with codecs.open(unigram_sentences_filepath, 'w',
                     encoding='utf_8') as f:
        for sentence in lemmatized_sentences_corpus(filename):
            f.write(sentence + '\n')


def save_bigram_sentences():
    """ Saves bigram sentences to file."""
    bigram_model = get_bigram_model()
    unigram_sentences = get_unigram_sentences()
    with codecs.open(bigram_sentences_filepath, 'w',
                     encoding='utf_8') as f:
        for sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[sentence])
            f.write(bigram_sentence + '\n')


def lemmatized_sentences_corpus(filename, remove_stop=False):
    nlp = spacy.load('en')
    for parsed_review in nlp.pipe(gen.load_csv(filename, 3),
                                  batch_size=10000, n_threads=4):
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token, remove_stop)])


def punct_space(token, remove_stop=False):
    if remove_stop:
        return token.is_punct or token.is_space or token.is_stop
    return token.is_punct or token.is_space


def get_bigram_model():
    model_exists = os.path.exists(bigram_model_filepath)
    if model_exists:
        logger.info("Loading bigram model from file")
        bigram_model = Phrases.load(bigram_model_filepath)
    else:
        logger.info("Creating new bigram model")
        unigram_sentences = get_unigram_sentences()
        bigram_model = Phrases(unigram_sentences)
        bigram_model.save(bigram_model_filepath)
    return bigram_model


def get_unigram_sentences():
    unigram_sentences = LineSentence(unigram_sentences_filepath)
    return unigram_sentences


def get_bigram_sentences():
    bigram_sentences = LineSentence(bigram_sentences_filepath)
    return bigram_sentences


def remove_punctuation(text):
    text = str(text)
    import re
    import string
    return re.sub('[' + string.punctuation + ']', '', text)
