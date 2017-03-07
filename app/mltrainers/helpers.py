"""Unthemed functions."""


def balance_data(df, column):
    """Function removing extra data to make proportion 50/50. (Only works for binary classes)."""
    thresh = df[column].value_counts().min()
    positive = df[df[column] == 1].sample(frac=1)
    negative = df[df[column] == -1].sample(frac=1)
    positive = positive[:thresh]
    negative = negative[:thresh]

    balanced = positive.append(negative)
    return balanced
