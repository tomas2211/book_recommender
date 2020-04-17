import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import editdistance

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


TOTAL_RATINGS_THRESH = 9
LIKE_THRESH = 5
SIGMA_MUL = 1
IMPLICIT_LIKES_AS_LIKE = False


isbns_fil = cnts[cnts > TOTAL_RATINGS_THRESH].index
print('Number of books with >%d ratings: %d' %(TOTAL_RATINGS_THRESH, len(isbns_fil)))
rat_exp = rat_explicit[rat_explicit.ISBN.isin(isbns_fil)]
rat_all = ratings[ratings.ISBN.isin(isbns_fil)]
books = books[books.ISBN.isin(isbns_fil)]

query_isbn = isbns_fil[531]
query_name = books[books.ISBN == query_isbn]['Book-Title'].item()

query_name = ''.lower()
distances = books['Book-Title'].apply(lambda x: editdistance.eval(x.lower(), query_name))
query = books[books.index == distances.idxmin()]

query_name = query['Book-Title'].item()
query_isbn = query['ISBN'].item()

print('Query book: %s' % query_name)
print('Query ratings: %d' % rat_exp[rat_exp.ISBN == query_isbn]['Book-Rating'].count())
print('Mean query rating: %.2f' % rat_exp[rat_exp.ISBN == query_isbn]['Book-Rating'].mean())

if IMPLICIT_LIKES_AS_LIKE:
    users_liking = rat_all[(rat_all.ISBN == query_isbn) &
                           ((rat_all['Book-Rating'] > LIKE_THRESH) | (rat_all['Book-Rating'] == 0))]['User-ID']
else:
    users_liking = rat_exp[(rat_exp.ISBN == query_isbn) & (rat_exp['Book-Rating'] > LIKE_THRESH)]['User-ID']
print('Users liking the query: %d' % len(users_liking))

relevant_rat = rat_exp[rat_exp['User-ID'].isin(users_liking)]

print('Relevant %d ratings.' % relevant_rat.shape[0])
print('Deciding among %d books.' % len(relevant_rat.ISBN.unique()))

relevant_rat_grouped = relevant_rat.groupby('ISBN')['Book-Rating']
cnt_ratings = relevant_rat_grouped.count()
mean_ratings = relevant_rat_grouped.mean()
if (cnt_ratings >= 2).sum() > 5:
    mean_ratings = mean_ratings[cnt_ratings > 2]
    std_ratings = relevant_rat_grouped.std()[cnt_ratings > 2]
    book_scores = mean_ratings - SIGMA_MUL*std_ratings
else:
    book_scores = mean_ratings

book_scores = book_scores.sort_values(ascending=False)


print('--- RECOMMENDING ---')
for i,r in pd.merge(book_scores, books, how='left', on='ISBN').head(5).iterrows():
    print('%.1f - %s' % (r['Book-Rating'], r['Book-Title']))


print('--- NOT RECOMMENDED ---')
for i,r in pd.merge(book_scores, books, how='left', on='ISBN').tail(5).iterrows():
    print('%.1f - %s' % (r['Book-Rating'], r['Book-Title']))
