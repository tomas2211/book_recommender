import argparse
import os
import textwrap

from tabulate import tabulate

from data_utils import load_ratings
from models.gmodel import Gmodel
from models.knn_model import KNNmodel
from models.n2vmodel import N2Vmodel
from models.pmodel import Pmodel
from util_scripts.isbn_finder import ISBN_Finder

test_queries = [
    '0671527215',  # Douglas Adams: Hitchhikers's Guide to the Galaxy
    '0767908473',  # ALLAN ZOLA KRONZEK: The Sorcerer's Companion: A Guide to the Magical World of Harry Potter
    '061815082X',  # J. R. R. Tolkien: The Hobbit (Young Adult edition, Sis cover)
    '0449213447',  # Chaim Potok: The Chosen
    '0156528207',  # Little Prince
    '0618002227'  # The fellowship of the ring
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    parser.set_defaults(interactive=False)
    parser.add_argument('--format', action='store_true')
    parser.set_defaults(format=False)
    args = parser.parse_args()

    if args.interactive:
        print('Loading...')

    ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()
    ratings_filtered = ratings[ratings.ISBN.isin(isbns_filtered)]
    ratings_train = ratings_filtered[~ratings_filtered['User-ID'].isin(test_useridx)]

    isbn_finder = ISBN_Finder('data/BX-Books.csv', isbns_filtered)

    models = [
        Pmodel(ratings_train),
        Gmodel(),
        N2Vmodel(),
        KNNmodel(ratings_train)
    ]


    def process_query(q_isbn):
        print('Query: {} - {}'.format(q_isbn, isbn_finder.isbn2name(q_isbn)))

        model_resp = []
        for m in models:
            output = m(q_isbn, ret_k=5, fill_to_k=True)
            model_resp.append([isbn_finder.isbn2name(isbn) for isbn in output])

        if args.format:
            try:
                width = (os.get_terminal_size().columns - 4) // 4 - 2
            except:
                width = (120 // 4) - 2
            resp_transp_warp = [[textwrap.fill(mod[i], width) for mod in model_resp] for i in range(5)]
            print(tabulate(resp_transp_warp, headers=['Pmodel', 'Gmodel', 'N2Vmodel', 'KNN model'], showindex="always"))
        else:
            for row in zip(*model_resp):
                print('\t'.join(row))


    if args.interactive:
        print('---------- INTERACTIVE MODE ----------')
        print('Ctrl+C/Write \'end\' to end.')
        if not args.format:
              print('Showing first five responses in columns: pmodel, gmodel, n2vmodel, knnmodel separated by tabs.')
        print('--------------------------------------')

        q_name = input('Query name: ')
        while q_name.lower() != 'end':
            q_isbn = isbn_finder.name2isbn(q_name)
            process_query(q_isbn)
            q_name = input('Query name: ')

    else:
        for q_isbn in test_queries:
            process_query(q_isbn)
