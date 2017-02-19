import csv


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
