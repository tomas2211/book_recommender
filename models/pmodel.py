import pandas as pd


class Pmodel:
    def __init__(
            self, ratings: pd.DataFrame,
            thresh_like_rating=5,
            sigma_mul=1.0,
            implicit_means_like=False
    ):
        self.thresh_like_rating = thresh_like_rating
        self.sigma_mul = sigma_mul
        self.implicit_means_like = implicit_means_like

        # Filter out explicit ratings (!= 0)
        self.rat_all = ratings
        self.rat_explicit = ratings[ratings['Book-Rating'] != 0]


    def __call__(self, query_isbn, ret_k=5, verbose=False, ret_scores=False, fill_to_k=False):
        # Extract all users that like the query
        if self.implicit_means_like:
            ratings_mask = (self.rat_all.ISBN == query_isbn) \
                           & (
                                   (self.rat_all['Book-Rating'] > self.thresh_like_rating)
                                   | (self.rat_all['Book-Rating'] == 0)
                           )
            users_liking = self.rat_all[ratings_mask]['User-ID']
        else:
            ratings_mask = (self.rat_explicit.ISBN == query_isbn) & (self.rat_explicit['Book-Rating'] >= self.thresh_like_rating)
            users_liking = self.rat_explicit[ratings_mask]['User-ID']

        if verbose:
            print('Users liking the query: %d' % len(users_liking))

        # Get the ratings from extracted users = relevant ratings
        rat_relevant = self.rat_explicit[self.rat_explicit['User-ID'].isin(users_liking)]

        if verbose:
            print('Relevant %d ratings.' % rat_relevant.shape[0])
            print('Deciding among %d books.' % len(rat_relevant.ISBN.unique()))

        # Process the relevant ratings
        rat_rel_grouped = rat_relevant.groupby('ISBN')['Book-Rating']
        ratings_cnt = rat_rel_grouped.count()
        ratings_mean = rat_rel_grouped.mean()

        # compute std if we can (i.e more than 2 relevant ratings) for at least K books
        if (ratings_cnt >= 2).sum() >= ret_k:
            ratings_mean = ratings_mean[ratings_cnt > 2]
            ratings_std = rat_rel_grouped.std()[ratings_cnt > 2]
            # subtract std to get minimal expected (on level of certainty) ratings
            book_scores = ratings_mean - self.sigma_mul * ratings_std
        else:
            book_scores = ratings_mean

        book_scores = book_scores.sort_values(ascending=False)
        book_scores = book_scores[book_scores.index != query_isbn]  # remove the query book if present

        if fill_to_k and len(book_scores) < ret_k:
            miss_n = ret_k - len(book_scores)
            # filling = pd.Series({'Book-Rating': [0] * miss_n})
            filling = self.rat_explicit.groupby('ISBN')['Book-Rating'].mean().sort_values(ascending=False)[:miss_n]
            book_scores = book_scores.append(filling)

        if ret_scores:
            return book_scores.index[:ret_k], book_scores.values[:ret_k]
        else:
            return book_scores.index[:ret_k]
