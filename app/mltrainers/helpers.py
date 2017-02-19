
def balance_data(df, column):
    thresh = df[column].value_counts().min()
    positive = df[df[column] == 1].sample(frac=1)
    negative = df[df[column] == -1].sample(frac=1)
    positive = positive[:thresh]
    negative = negative[:thresh]

    balanced = positive.append(negative)
    return balanced
