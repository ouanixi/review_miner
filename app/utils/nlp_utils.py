import codecs
import os.path
import re
import string

from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from app.utils import generators as gen
from config.app.default import *

unigram_sentences_filepath = UNIGRAM_SENTENCES
bigram_sentences_filepath = BIGRAM_SENTENCES
bigram_model_filepath = BIGRAM_MODEL
RAW_REVIEWS = RAW_REVIEWS


def make_corpus():
    """ Takes in a path for the raw corpus and creates
        two news corpora:
        1- unigram sentences
        2- bigram sentences
    """
    # save_unigram_sentences(RAW_REVIEWS)
    save_bigram_sentences()


def make_sentences(paragraph):
    parsed_review = gen.NLP().nlp(paragraph)
    for sent in parsed_review.sents:
        yield sent.text.lower()


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
    for parsed_review in gen.NLP().nlp.pipe(gen.load_csv(filename, 3),
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
        bigram_model = Phrases.load(bigram_model_filepath)
    else:
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
    return re.sub('[' + string.punctuation + ']', '', text)
