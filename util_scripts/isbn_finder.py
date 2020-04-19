import pandas as pd
from fuzzywuzzy import fuzz

from data_utils import load_ratings


def find_book_isbn(name: str, books: pd.DataFrame):
    name = name.lower()
    distances = books['Book-Title'].apply(lambda x: fuzz.partial_ratio(x.lower(), name))
    return distances.sort_values().index[-1:-5:-1].array


df_books = pd.read_csv(
    '../data/BX-Books.csv',
    usecols=['ISBN', 'Book-Title', 'Book-Author'],
    sep=';',
    encoding='latin-1',
    index_col='ISBN'
)

ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

# Filter ratings to explicit and by isbns_filtered
rat_explicit = ratings[ratings['Book-Rating'] != 0]
rat_exp = rat_explicit[rat_explicit.ISBN.isin(isbns_filtered)]
rat = ratings[ratings.ISBN.isin(isbns_filtered)]

df_books_fil = df_books[df_books.index.isin(isbns_filtered)]

name = input('Enter book name:')
while name.lower() != 'end':
    isbns = find_book_isbn(name, df_books_fil)

    for isbn in isbns:
        isbn_rat = rat_exp[rat_exp.ISBN == isbn]
        rat_num = len(isbn_rat)
        meanrat = isbn_rat['Book-Rating'].mean()

        q_name = df_books.loc[isbn]['Book-Title']
        q_author = df_books.loc[isbn]['Book-Author']

        print('Found: %s: %s - %s; mean rating %.2f (%d)' % (q_author, q_name, isbn, meanrat, rat_num))

    name = input('Enter book name:')
