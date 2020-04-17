import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import editdistance
from pmodel import Pmodel


def find_book_isbn(name: str, books: pd.DataFrame):
    name = name.lower()
    # Calculate levhenstein distances
    distances = books['Book-Title'].apply(lambda x: editdistance.eval(x.lower(), name))
    return distances.idxmin()


ratings = pd.read_csv(
    'data/BX-Book-Ratings.csv',
    usecols=['User-ID', 'ISBN', 'Book-Rating'],
    dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'int32'},
    sep=';'
)

books = pd.read_csv(
    'data/BX-Books.csv',
    usecols=['ISBN', 'Book-Title', 'Book-Author'],
    sep=';',
    encoding='latin-1',
    index_col='ISBN'
)

cnts = ratings['ISBN'].value_counts()
plt.bar(np.arange(1000), cnts[:1000], width=1.0)
plt.title('Rating (implicit) count distribution')
plt.xlabel('Book index (sorted by number of ratings)')
plt.ylabel('Number of ratings')
plt.show()

cnts = ratings[ratings['Book-Rating'] != 0]['ISBN'].value_counts()
plt.bar(np.arange(1000), cnts[:1000], width=1.0)
plt.title('Rating (explicit) count distribution')
plt.xlabel('Book index (sorted by number of ratings)')
plt.ylabel('Number of ratings')
plt.show()

pmodel = Pmodel(
    ratings,
    filter_books_numratings=9,
    thresh_like_rating=5,
    sigma_mul=1.0,
    implicit_means_like=True,
    verbose=True
)

q_isbn = find_book_isbn('The chosen', books)
q_name = books.loc[q_isbn]['Book-Title']
q_author = books.loc[q_isbn]['Book-Author']

print('Found: %s - %s (%s)' % (q_isbn, q_name, q_author))
print('Query ratings: %d' % ratings[ratings.ISBN == q_isbn]['Book-Rating'].count())
print('Mean query rating (explicit): %.2f' %
      ratings[(ratings.ISBN == q_isbn) & (ratings['Book-Rating'] != 0)]['Book-Rating'].mean())

resp_isbns, resp_scores = pmodel(q_isbn, verbose=True, ret_results=5)

selection = books.loc[resp_isbns]

print('--- RECOMMENDING ---')
for bk, sc in zip(selection.iterrows(), resp_scores):
    print('%.2f - %s' % (sc, bk[1]['Book-Title']))


