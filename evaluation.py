import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pandarallel import pandarallel
from tqdm import tqdm

from data_utils import load_ratings
from models.gmodel import Gmodel
from models.knn_model import KNNmodel
from models.n2vmodel import N2Vmodel
from models.pmodel import Pmodel


def evaluate(model, args, test_useridx, ratings_test):
    dcg_weights = 1 / np.log2(np.arange(2, args.K + 2))
    users_ndcg = []
    users_ndcg_wh = []
    users_ndcg_max = []
    eval_record = {}

    tested_books_tot = 0
    unrated_books_tot = 0

    for user in tqdm(test_useridx):
        rats_usr = ratings_test[ratings_test['User-ID'] == user]  # users ratings

        # ndcg normalization
        best_res = rats_usr['Book-Rating'].sort_values(ascending=False)[:args.K].to_numpy().astype(np.float)
        dcg_best = best_res * dcg_weights

        liked_books = rats_usr[rats_usr['Book-Rating'] >= args.like_threshold]['ISBN']  # get liked books
        rats_usr = rats_usr.set_index('ISBN')

        def eval(x):  # evaluate on model
            return model(x, ret_k=args.K, fill_to_k=True)

        outputs = liked_books.parallel_apply(eval)  # Parallelize the evaluation

        # rate the outputs by user rating
        out_rated_pd = outputs.apply(lambda out: rats_usr.reindex(out, fill_value=0)['Book-Rating'].values)
        out_rated = np.vstack(out_rated_pd.array).astype(np.float)

        tested_books = len(liked_books)  # number of tested books
        unrated_books_tot += np.sum(out_rated == 0)
        tested_books_tot += tested_books

        # calc standard ndcg
        ndcg = np.sum(out_rated * dcg_weights, axis=1) / np.sum(dcg_best)
        users_ndcg.append(ndcg)

        # ndcg with skipping unrated books
        mask = out_rated != 0
        best_w = dcg_best[None, :].repeat(tested_books, 0)
        best_w[~mask] = 0.0
        ndcg_wh = np.sum(out_rated * dcg_weights, axis=1) / (np.sum(best_w, axis=1) + 1e-10)
        ndcg_wh = ndcg_wh[np.any(mask, axis=1)]
        users_ndcg_wh.append(ndcg_wh)

        # maxndcg per book
        ndcg_m = np.max(out_rated * dcg_weights, axis=1) / np.max(dcg_best)
        users_ndcg_max.append(ndcg_m)

        eval_record[user] = {
            'liked_books': liked_books,
            'outputs': outputs,
            'outputs_rated': out_rated,
            'ndcg': ndcg,
            'ndcg_wh': ndcg_wh,
            'ndcg_max': ndcg_m
        }

    eval_record['tested_books_tot'] = tested_books_tot
    eval_record['unrated_books_tot'] = unrated_books_tot
    eval_record['users_ndcg'] = users_ndcg
    eval_record['users_ndcg_wh'] = users_ndcg_wh
    eval_record['users_ndcg_max'] = users_ndcg_max

    return eval_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', default='eval', help="Folder to store evaluation files.")
    parser.add_argument('--model', required=True, type=str, help="Model to evaluate [pmodel|gmodel|n2vmodel|knnmodel]")
    parser.add_argument('--K', required=True, type=int, help="Number of queried results")
    parser.add_argument('--like_threshold', default=5, type=float, help="Rating threshold for \'liked\' books")

    parser.add_argument('--pmodel_implicit_is_like', action='store_true',
                        help='Pmodel parameter - implicit rating means user likes book.')
    parser.set_defaults(pmodel_implicit_is_like=False)
    parser.add_argument('--pmodel_sigmul', default=0.1, type=float, help='Pmodel parameter - sigma multiplier')

    parser.add_argument('--gmodel_graphpath', default='data/corat_graph_18-04-2020_1655',
                        help='Gmodel parameter - path to processed graph')
    parser.add_argument('--gmodel_sigmul', default=0.1, type=float, help='Gmodel parameter - sigma multiplier')
    parser.add_argument('--gmodel_robdist_clip', default=40, type=float, help='Gmodel parameter - distance clipping')
    parser.add_argument('--gmodel_minimal_corats', default=3, type=float,
                        help='Gmodel parameter - minimal number of coratings')

    parser.add_argument('--n2vmodel_embed_fn', default='data/node2vec_dict', type=str,
                        help='N2V model parameter - path to embeddings')

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    pandarallel.initialize()

    # Load data
    ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

    ratings_filtered = ratings[ratings.ISBN.isin(isbns_filtered)]
    rat_explicit = ratings_filtered[ratings_filtered['Book-Rating'] != 0]

    ratings_train = ratings_filtered[~ratings_filtered['User-ID'].isin(test_useridx)]
    ratings_test = ratings_filtered[ratings_filtered['User-ID'].isin(test_useridx)]

    # Load model
    if args.model == 'pmodel':
        model = Pmodel(
            ratings_train,
            thresh_like_rating=args.like_threshold,
            sigma_mul=args.pmodel_sigmul,
            implicit_means_like=args.pmodel_implicit_is_like
        )
    elif args.model == 'gmodel':
        model = Gmodel(
            graph_path=args.gmodel_graphpath,
            sigma_mul=args.gmodel_sigmul, robdist_clip=args.gmodel_robdist_clip,
            minimal_corats=args.gmodel_minimal_corats
        )
    elif args.model == 'n2vmodel':
        model = N2Vmodel(embed_fn=args.n2vmodel_embed_fn)
    elif args.model == 'knnmodel':
        model = KNNmodel(ratings_train)
    else:
        raise RuntimeError('Invalid model specified: {}'.format(args.model))

    # Load/create eval record
    eval_rec_fn = os.path.join(args.out_folder, 'eval_record.pickle')

    if not os.path.exists(eval_rec_fn):
        # Evaluate if no record exists
        eval_record = evaluate(model, args, test_useridx, ratings_test)
        with open(eval_rec_fn, 'wb') as f:
            pickle.dump(eval_record, f)
    else:
        with open(eval_rec_fn, 'rb') as f:
            eval_record = pickle.load(f)

    # Create histograms and store the results
    users_ndcg = eval_record['users_ndcg']
    users_ndcg_wh = eval_record['users_ndcg_wh']
    users_ndcg_max = eval_record['users_ndcg_max']
    tested_books_tot = eval_record['tested_books_tot']
    unrated_books_tot = eval_record['unrated_books_tot']

    users_ndcg = np.hstack(users_ndcg)
    plt.hist(users_ndcg, bins=100)
    plt.title('NDCG histogram')
    plt.xlabel('NDCG')
    plt.ylabel('# of queries')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_folder, 'ndcg_hist.png'))
    plt.close()

    users_ndcg_wh = np.hstack(users_ndcg_wh)
    plt.hist(users_ndcg_wh, bins=100)
    plt.title('NDCG-ignore-unevaluated histogram')
    plt.xlabel('NDCG-ignore-unevaluated')
    plt.ylabel('# of queries')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_folder, 'ndcg-iu_hist.png'))
    plt.close()

    users_ndcg_max = np.hstack(users_ndcg_max)
    plt.hist(users_ndcg_max, bins=100)
    plt.title('NDCG-max histogram')
    plt.xlabel('NDCG-max')
    plt.ylabel('# of queries')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_folder, 'ndcg-max_hist.png'))
    plt.close()

    ndcg = np.mean(users_ndcg)
    ndcgm = np.mean(users_ndcg_max)
    ndcgwh = np.mean(users_ndcg_wh)

    print('Tested books: %d' % tested_books_tot)
    print('Missing ratings total: %d' % unrated_books_tot)
    print('Unrated fraction %.3f' % (unrated_books_tot / (tested_books_tot * args.K)))
    print('NDCG: %.3f' % ndcg)
    print('NDCG-M: %.3f' % ndcgm)
    print('NDCG-IU: %.3f' % ndcgwh)

    with open(os.path.join(args.out_folder, 'stats'), 'w') as f:
        f.write('Tested books: %d\n' % tested_books_tot)
        f.write('Missing ratings total: %d\n' % unrated_books_tot)
        f.write('Unrated fraction %.3f\n' % (unrated_books_tot / (tested_books_tot * args.K)))
        f.write('NDCG: %.3f\n' % ndcg)
        f.write('NDCG-M: %.3f\n' % ndcgm)
        f.write('NDCG-IU: %.3f' % ndcgwh)
