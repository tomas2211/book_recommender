import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class KNNmodel:
    def __init__(self, ratings: pd.DataFrame):
        pivot_tab = ratings.pivot(
            index='ISBN',
            columns='User-ID',
            values='Book-Rating'
        ).fillna(0)
        self.isbn_indexes = pivot_tab.index
        self.book_feats = csr_matrix(pivot_tab.values)

        model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=1)
        self.model = model.fit(self.book_feats)

    def __call__(self, query_isbn, ret_k=5, verbose=False, ret_scores=False, fill_to_k=False):
        isbn_idx = self.isbn_indexes.get_loc(query_isbn)
        q_feats = self.book_feats[isbn_idx, :]

        out_scores, out_idxs = self.model.kneighbors(q_feats, ret_k + 1)

        out_idxs = out_idxs[0, 1:]
        outs_isbns = self.isbn_indexes[out_idxs].values

        if ret_scores:
            ret_scores = ret_scores[0, 1:]
            return outs_isbns, ret_scores
        return outs_isbns
