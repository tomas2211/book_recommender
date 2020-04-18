import pandas as pd
import pickle


def load_ratings():
    with open('data/testtrain.pickle', 'rb') as f:
        ld = pickle.load(f)
        isbns_filtered = ld['isbns_filtered']
        test_useridx = ld['test_useridx']
        train_useridx = ld['train_useridx']

    ratings = pd.read_csv(
        'data/BX-Book-Ratings.csv',
        usecols=['User-ID', 'ISBN', 'Book-Rating'],
        dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'int32'},
        sep=';',
        encoding='latin-1'
    )
    return ratings, isbns_filtered, test_useridx, train_useridx