import editdistance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_book_isbn(name: str, books: pd.DataFrame):
    name = name.lower()
    # Calculate levhenstein distances
    distances = books['Book-Title'].apply(lambda x: editdistance.eval(x.lower(), name))
    return distances.idxmin()


ratings = pd.read_csv(
    '../data/BX-Book-Ratings.csv',
    usecols=['User-ID', 'ISBN', 'Book-Rating'],
    dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'int32'},
    sep=';',
    encoding='latin-1'
)

books = pd.read_csv(
    '../data/BX-Books.csv',
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
plt.tight_layout()
plt.savefig('vis/dataset_implicit_rats.png')
plt.close()

cnts = ratings[ratings['Book-Rating'] != 0]['ISBN'].value_counts()
plt.bar(np.arange(1000), cnts[:1000], width=1.0)
plt.title('Rating (explicit) count distribution')
plt.xlabel('Book index (sorted by number of ratings)')
plt.ylabel('Number of ratings')
plt.tight_layout()
plt.savefig('vis/dataset_explicit_rats.png')
plt.close()
