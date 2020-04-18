import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from pmodel import Pmodel
from tqdm import tqdm
import pickle
from data_utils import load_ratings

LIKE_RATING = 5
K = 10

ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

ratings_filtered = ratings[ratings.ISBN.isin(isbns_filtered)]
rat_explicit = ratings_filtered[ratings_filtered['Book-Rating'] != 0]

ratings_train = ratings_filtered[~ratings_filtered['User-ID'].isin(test_useridx)]
ratings_test = ratings_filtered[ratings_filtered['User-ID'].isin(test_useridx)]

pmodel = Pmodel(
    ratings_train,
    thresh_like_rating=5,
    sigma_mul=1.0,
    implicit_means_like=False,
)

dcg_weights = 1 / np.log2(np.arange(2, K+2))
users_ndcg = []
users_ndcg_wh = []

tested_books = 0
unrated = 0

for user in tqdm(test_useridx):
    rats_usr = ratings_test[ratings_test['User-ID'] == user]

    best_res = rats_usr['Book-Rating'].sort_values(ascending=False)[:K].to_numpy().astype(np.float)
    dcg_best = best_res * dcg_weights

    liked_books = rats_usr[rats_usr['Book-Rating'] > LIKE_RATING]['ISBN']

    rats_usr = rats_usr.set_index('ISBN')

    outputs = liked_books.apply(lambda x: pmodel(x, ret_k=K, fill_to_k=True))
    out_tab = outputs.apply(lambda out: rats_usr.reindex(out, fill_value=0)['Book-Rating'].values)
    out_rated = np.vstack(out_tab.array).astype(np.float)

    tbooks = out_rated.shape[0]

    unrated += np.sum(out_rated == 0)
    tested_books += tbooks

    ndcg = np.sum(out_rated * dcg_weights, axis=1) / np.sum(dcg_best)

    mask = out_rated != 0
    best_w = dcg_best[None, :].repeat(tbooks, 0)
    best_w[~mask] = 0.0
    ndcg_wh = np.sum(out_rated * dcg_weights, axis=1) / np.sum(best_w, axis=1)
    ndcg_wh = ndcg_wh[~np.isnan(ndcg_wh)]

    users_ndcg.append(ndcg)
    users_ndcg_wh.append(ndcg_wh)

users_ndcg = np.hstack(users_ndcg)
np.mean(users_ndcg)
plt.hist(users_ndcg, bins=100)
plt.show()

users_ndcg_wh = np.hstack(users_ndcg_wh)
np.mean(users_ndcg_wh)
plt.hist(users_ndcg_wh, bins=100)
plt.show()


unrated / (tested_books * K)
