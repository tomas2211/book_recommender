import matplotlib.pyplot as plt
import numpy as np
from models.pmodel import Pmodel
from tqdm import tqdm
import pickle
from data_utils import load_ratings
from models.gmodel import Gmodel
from models.n2vmodel import N2Vmodel
from models.knn_model import KNNmodel
from pandarallel import pandarallel
import argparse
import os
import pandas as pd


# From ISBN finder:
#  Douglas Adams: Hitchhikers's Guide to the Galaxy - 0671527215; mean rating 8.58 (26)
#  ALLAN ZOLA KRONZEK: The Sorcerer's Companion: A Guide to the Magical World of Harry Potter - 0767908473; mean rating 7.91 (11)
#  J. R. R. Tolkien: The Hobbit (Young Adult edition, Sis cover) - 061815082X; mean rating 8.23 (13)
#  Chaim Potok: The Chosen - 0449213447; mean rating 8.12 (26)

test_queries = [
    # '0671527215',
    # '0767908473',
    # '061815082X',
    # '0449213447',
    '0156528207',  # Little Prince
    '0618002227'  # The fellowship...
]

ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()
ratings_filtered = ratings[ratings.ISBN.isin(isbns_filtered)]
ratings_train = ratings_filtered[~ratings_filtered['User-ID'].isin(test_useridx)]

df_books = pd.read_csv(
    'data/BX-Books.csv',
    usecols=['ISBN', 'Book-Title', 'Book-Author'],
    sep=';',
    encoding='latin-1',
    index_col='ISBN'
)

models = [
    Pmodel(ratings_train, sigma_mul=0.1, implicit_means_like=False, thresh_like_rating=5),
    Gmodel(sigma_mul=0.1, minimal_corats=3),
    N2Vmodel(),
    KNNmodel(ratings_train)
]

for q_isbn in test_queries:
    print('Query: {}'.format(q_isbn))

    model_resp = []
    for m in models:
        out = m(q_isbn, ret_k=5, fill_to_k=True)
        model_resp.append(
            [
                '{}: {}'.format(auth, tit) for auth, tit in
                zip(df_books.reindex(out)['Book-Author'], df_books.reindex(out)['Book-Title'])
            ]
        )


    for row in zip(*model_resp):
        print('\t'.join(row))



