import csv

import spacy


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class NLP(metaclass=Singleton):
    def __init__(self):
        self.nlp = spacy.load('en')


def load_csv(csv_file, col):
    """Generates values of a given column in a csv file lazily.

    Keyword arguments:
    csv_file -- path to the csv file
    col -- column to read data from
    """
    with open(csv_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            yield (row[col])
