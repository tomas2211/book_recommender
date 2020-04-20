import pickle

import pandas as pd
from fuzzywuzzy import fuzz


class KNNmodel:
    def __init__(self):
        with open('knnmodel.pickle', 'rb') as f:
            dat = pickle.load(f)

        self.model = dat['model']
        self.isbn_indexes = dat['isbn_indexes']
        self.book_feats = dat['book_feats']

    def __call__(self, query_isbn, ret_k=5, verbose=False, ret_scores=False, fill_to_k=False):
        isbn_idx = self.isbn_indexes.get_loc(query_isbn)
        q_feats = self.book_feats[isbn_idx, :]

        out_scores, out_idxs = self.model.kneighbors(q_feats, ret_k + 1)

        out_idxs = out_idxs[0, 1:]
        outs_isbns = self.isbn_indexes[out_idxs].values

        if ret_scores:
            out_scores = out_scores[0, 1:]
            return list(outs_isbns), list(out_scores)
        return outs_isbns


class isbn2name:
    def __init__(self):
        self.df_books = pd.read_csv('book_names.csv', index_col='ISBN')

    def isbn2name(self, isbn):
        try:
            resp = self.df_books.loc[isbn]
        except KeyError:
            return None

        return '{}: {}'.format(resp['Book-Author'], resp['Book-Title'])

    def name2isbn(self, name):
        name = name.lower()
        distances = self.df_books['Book-Title'].apply(lambda x: fuzz.ratio(str(x).lower(), name))
        return distances.idxmax()
