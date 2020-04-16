import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ratings = pd.read_csv('data/BX-Book-Ratings.csv',
                    usecols=['User-ID', 'ISBN', 'Book-Rating'],
                    dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'int32'},
                    sep=';',
                    encoding='latin-1'
                    )

books = pd.read_csv('data/BX-Books.csv',
                    usecols=['ISBN', 'Book-Title'],
                    sep=';',
                    encoding='latin-1'
                    )

rat_explicit = ratings[ratings['Book-Rating'] != 0]

cnts = rat_explicit['ISBN'].value_counts()

plt.bar(np.arange(1000), cnts[:1000], width=1.0)
plt.title('Rating count distribution')
plt.xlabel('Book index (sorted by number of ratings)')
plt.ylabel('Number of ratings')
plt.show()


RATING_NO_THRESH = 9
LIKE_THRESH = 5

isbns_fil = cnts[cnts > RATING_NO_THRESH].index
print('Number of books with >%d ratings: %d' %(RATING_NO_THRESH, len(isbns_fil)))
rat = rat_explicit[rat_explicit.ISBN.isin(isbns_fil)]

query_isbn = '0740718398' #isbns_fil[128714]
query_name = books[books.ISBN == query_isbn]['Book-Title'].item()
print('Query book: %s' % query_name)


users_liking = rat[(rat.ISBN == query_isbn) & (rat['Book-Rating'] > LIKE_THRESH)]['User-ID']
print('Users liking the book: %d' % len(users_liking))

relevant_rat = rat[rat['User-ID'].isin(users_liking)]

print('Relevant %d ratings.' % relevant_rat.shape[0])
print('Deciding among %d books.' % len(relevant_rat.ISBN.unique()))

cnt_ratings = relevant_rat.groupby('ISBN')['Book-Rating'].count()
mean_ratings = relevant_rat.groupby('ISBN')['Book-Rating'].mean()#[cnt_ratings > 2]
std_ratings = relevant_rat.groupby('ISBN')['Book-Rating'].std()#[cnt_ratings > 2]

minimal_expected = (mean_ratings - 1*std_ratings).sort_values(ascending=False)

for i,r in pd.merge(minimal_expected, books, how='left', on='ISBN').head(5).iterrows():
    print('%.1f - %s' % (r['Book-Rating'], r['Book-Title']))

for i,r in pd.merge(minimal_expected, books, how='left', on='ISBN').tail(5).iterrows():
    print('%.1f - %s' % (r['Book-Rating'], r['Book-Title']))
