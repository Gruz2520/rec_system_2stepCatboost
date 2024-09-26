import json
from collections import Counter


class ColdStartRecommender:
    def __init__(self, implementation='global', local_logs_df=None):
        self.implementation = implementation
        self.local_logs_df = local_logs_df

    def recommend(self, logs_df, movies_df, user_id, top_n=10):
        user_watched_films = logs_df[logs_df.user_id == user_id].movie_id.unique()

        watched_movies = movies_df[movies_df.id.isin(user_watched_films)]
        unwatched_movies = movies_df[~movies_df.id.isin(user_watched_films)]

        watched_genres = []

        for movie in watched_movies.iterrows():
            watched_genres.extend(json.loads(movie[1].genres))

        favourite_genres = [genre_id for genre_id, _ in Counter(watched_genres).most_common(10)]

        if len(favourite_genres):
            top_genres_movies = unwatched_movies[unwatched_movies.genres.apply(lambda x: len(set(json.loads(x)) & set(favourite_genres)) > 0)]

            if len(set(self.local_logs_df.index) & set(top_genres_movies.id.unique())) >= top_n:
                unwatched_movies = top_genres_movies

        if self.implementation == 'global':
            return unwatched_movies.sort_values('views', ascending=False).head(top_n).id.values
        elif self.implementation == 'local':
            local_top = self.local_logs_df[self.local_logs_df.index.isin(unwatched_movies.id.unique())]
            return list(local_top.head(top_n).index.values.astype(int))
        else:
            raise KeyError
