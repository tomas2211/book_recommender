import pandas as pd
from fuzzywuzzy import fuzz


class ISBN_Finder():
    def __init__(self, books_csv_fn='data/BX-Books.csv', isbns_filter=None):
        self.df_books = pd.read_csv(
            books_csv_fn,
            usecols=['ISBN', 'Book-Title', 'Book-Author'],
            sep=';',
            encoding='latin-1',
            index_col='ISBN'
        )

        if isbns_filter is not None:
            self.df_books = self.df_books[self.df_books.index.isin(isbns_filter)]

    def _match_name(self, name: str):
        name = name.lower()
        return self.df_books['Book-Title'].apply(lambda x: fuzz.partial_ratio(x.lower(), name))

    def name2isbn_top5(self, name: str):
        return self._match_name(name).sort_values().index[-1:-6:-1].array

    def name2isbn(self, name: str):
        return self._match_name(name).idxmax()

    def isbn2name(self, isbn: str, softerror=True):
        try:
            return '{}: {}'.format(self.df_books.loc[isbn]['Book-Author'], self.df_books.loc[isbn]['Book-Title'])
        except KeyError:
            if softerror:
                return '<N/A>: <N/A>'
            else:
                return None


if __name__ == '__main__':
    from data_utils import load_ratings

    ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

    isbn_finder = ISBN_Finder(isbns_filter=isbns_filtered)

    # Filter ratings to explicit and by isbns_filtered
    rat_exp = ratings[ratings['Book-Rating'] != 0]
    rat_exp = rat_exp[rat_exp.ISBN.isin(isbns_filtered)]

    name = input('Enter book name:')
    while name.lower() != 'end':
        isbns = isbn_finder.name2isbn_top5(name)
        for isbn in isbns:
            isbn_rat = rat_exp[rat_exp.ISBN == isbn]
            rat_num = len(isbn_rat)
            meanrat = isbn_rat['Book-Rating'].mean()
            found_name = isbn_finder.isbn2name(isbn)
            print('Found: %s - %s; mean rating %.2f (%d)' % (found_name, isbn, meanrat, rat_num))

        name = input('Enter book name:')
