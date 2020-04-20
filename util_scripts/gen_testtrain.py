import pandas as pd
import pickle

BOOK_MINIMAL_NRAT = 10  # explicit ratings
TESTUSER_MINIMAL_NRAT = 50
NTEST_USERS = 40

ratings = pd.read_csv(
    '../data/BX-Book-Ratings.csv',
    usecols=['User-ID', 'ISBN', 'Book-Rating'],
    dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'int32'},
    sep=';',
    encoding='latin-1'
)

# Filter books by number of explicit ratings (thresh_min_ratings)
rat_explicit = ratings[ratings['Book-Rating'] != 0]
cnts = rat_explicit['ISBN'].value_counts()

isbns_filtered = cnts[cnts > BOOK_MINIMAL_NRAT].index
rat_explicit = rat_explicit[rat_explicit.ISBN.isin(isbns_filtered)]

# Filter users by number of ratings
rat_cnts = rat_explicit['User-ID'].value_counts()
rat_cnts_fil = rat_cnts[rat_cnts > TESTUSER_MINIMAL_NRAT]

print('Number of books with >%d explicit ratings: %d' % (BOOK_MINIMAL_NRAT, len(isbns_filtered)))
print('Users with >%d explicit ratings: %d' % (TESTUSER_MINIMAL_NRAT, len(rat_cnts_fil)))

# select test users and save the rest
test_useridx = rat_cnts_fil.sample(NTEST_USERS, random_state=211, replace=False).index.array
train_useridx = rat_cnts[~rat_cnts.index.isin(test_useridx)].index.array

with open('../data/testtrain.pickle', 'wb') as f:
    pickle.dump(
        {
        'isbns_filtered': isbns_filtered,
        'test_useridx': test_useridx,
        'train_useridx': train_useridx
        },
        f
    )

# Export filtered ISBNs for GCP app

df_books = pd.read_csv(
    'data/BX-Books.csv',
    usecols=['ISBN', 'Book-Title', 'Book-Author'],
    sep=';',
    encoding='latin-1',
    index_col='ISBN'
)

df_books.loc[isbns_filtered].to_csv('gcp_app/book_names.csv')