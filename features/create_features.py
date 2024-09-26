import numpy as np
import pandas as pd


class FeatureCreator:
    def __init__(self, production_version: bool, n_weeks_for_catboost: int) -> None:
        self.production_version = production_version
        self.n_weeks_for_catboost = n_weeks_for_catboost

    @staticmethod
    def activation_function(x):
        if x < 1/2:
            return 1 / (1 + np.exp(-10 * (x - 1/2)))
        else:
            return 1 / (1 + np.exp(-3.6 * (x - 1/2)))
        
    @staticmethod
    def get_movie_rating(x):
        return np.ceil(x * 5)

    @staticmethod
    def smooth(series, window_size, smoothing_func):
        series = np.array(series)
        ext = np.r_[
            2 * series[0] - series[window_size - 1::-1],
            series,
            2 * series[-1] - series[-1:-window_size:-1],
        ]
        weights = smoothing_func(window_size)
        smoothed = np.convolve(weights / weights.sum(), ext, mode="same")
        return smoothed[window_size:-window_size + 1]

    def trend_slope(self, series, window_size=7, smoothing_func=np.hamming):
        smoothed = self.smooth(series, window_size, smoothing_func)
        return smoothed[-1] - smoothed[-2]

    def make_trend_slope_with_watches_in_time(self, logs_df, movies_df, window_time=7):
        item_stats = movies_df[["id"]].set_index("id")
        max_date = logs_df["day_of_year"].max()
        cols = list(range(window_time))
        for col in cols:
            watches = logs_df[
                logs_df["day_of_year"] == max_date - (window_time - 1 - col)
                ]
            item_stats = item_stats.join(
                watches.groupby("movie_id")["user_id"].count(), lsuffix=col
            )
        item_stats.fillna(0, inplace=True)
        new_colnames = ["user_id" + str(i) for i in range(1, window_time)] + ["user_id"]
        trend_slope_to_row = lambda row: self.trend_slope(row[new_colnames], window_size=window_time)
        item_stats["trend_slope_in_7_days"] = item_stats.apply(trend_slope_to_row, axis=1)
        item_stats["watched_in_7_days"] = item_stats[new_colnames].apply(sum, axis=1)

        if 'trend_slope_in_7_days' in movies_df.columns:
            movies_df = movies_df.drop(["trend_slope_in_7_days", "watched_in_7_days"], axis=1)

        movies_df = movies_df.merge(
            item_stats[["trend_slope_in_7_days", "watched_in_7_days"]], left_on='id',
            right_index=True)

        return movies_df

    @staticmethod
    def count_user_total_watch_duration(logs_df):
        sum_duration = pd.DataFrame(logs_df.groupby(['user_id', 'movie_id'])['duration'].sum())
        sum_duration.reset_index(inplace=True)

        return sum_duration

    @staticmethod
    def get_approximate_mean_duration(sum_df):
        approx_duration = sum_df.groupby('movie_id')['duration'].quantile(0.9)
        mean_duration = sum_df.groupby('movie_id')['duration'].mean()

        return approx_duration, mean_duration

    @staticmethod
    def add_duration_features_in_movies(movies_df, approx_duration, mean_duration):
        movies_df['approx_duration'] = movies_df['id'].apply(lambda x: approx_duration[x] if x in approx_duration else np.nan)
        movies_df['mean_duration'] = movies_df['id'].apply(lambda x: mean_duration[x] if x in mean_duration else np.nan)
        movies_df['mean_percentage'] = movies_df['mean_duration'] / movies_df['approx_duration'] * 100

        return movies_df

    def add_duration_features_in_logs(self, logs_df, movies_df):
        logs_df['percent_of_watch'] = np.zeros(logs_df.shape[0])
        movies_ids = movies_df.id.unique().tolist()

        for id in movies_ids:
            logs_df.loc[logs_df.movie_id == id, 'percent_of_watch'] = (logs_df.loc[logs_df.movie_id == id, 'duration'] / \
                                                                       movies_df.loc[
                                                                           movies_df.id == id, 'approx_duration'].values[0])

        logs_df['score'] = logs_df['percent_of_watch'].apply(self.activation_function)
        logs_df['rating'] = logs_df['score'].apply(self.get_movie_rating)

        return logs_df

    @staticmethod
    def add_score_features_in_movies(logs_df, movies_df):
        movies_scores = logs_df.groupby('movie_id').agg({'score': ['mean', 'median', 'std', lambda x: x.quantile(0.25),
                                                                   lambda x: x.quantile(0.75)]}).sort_index()
        ratings_count = logs_df.groupby('movie_id')['rating'].value_counts(normalize=True).unstack().sort_index().fillna(0)

        movies_scores.columns = ['mean_score', 'median_score', 'std_score', 'q25_score', 'q75_score']
        ratings_count.columns = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']

        movies_df = movies_df.merge(movies_scores, left_on='id', right_index=True, how='outer')
        movies_df = movies_df.merge(ratings_count, left_on='id', right_index=True, how='outer')

        return movies_df

    def make_dataset_split(self, logs_df):
        if self.production_version:
            d1, d2 = logs_df[logs_df.week_number < logs_df.week_number.max() - self.n_weeks_for_catboost], logs_df[
                logs_df.week_number >= logs_df.week_number.max() - self.n_weeks_for_catboost]
            d2 = d2[d2.user_id.isin(d1.user_id.unique())]
            # d1 = d1[d1.user_id.isin(d2.user_id.unique())]

            return d1, d2, None
        else:
            df, test_test = logs_df[logs_df.week_number < logs_df.week_number.max() - 2], logs_df[logs_df.week_number >= logs_df.week_number.max() - 2]
            d1, d2 = df[df.week_number < df.week_number.max() - self.n_weeks_for_catboost], df[df.week_number >= df.week_number.max() - self.n_weeks_for_catboost]
            d2 = d2[d2.user_id.isin(d1.user_id.unique())]
            return d1, d2, test_test

    def add_duration_features_in_train_test(self, train_df, test_df, movies_df):
        train_df = self.add_duration_features_in_logs(train_df, movies_df)
        movies_df = self.add_score_features_in_movies(train_df, movies_df)
        test_df = self.add_duration_features_in_logs(test_df, movies_df)

        return train_df, test_df, movies_df

    def make_features(self, dataset):
        sum_duration = self.count_user_total_watch_duration(dataset.logs)
        approx_duration, mean_duration = self.get_approximate_mean_duration(sum_duration)

        dataset.movies = self.add_duration_features_in_movies(dataset.movies, approx_duration, mean_duration)

        return dataset

    def split(self, dataset):
        train_df, test_df, test_test = self.make_dataset_split(dataset.logs)
        train_df, test_df, dataset.movies = self.add_duration_features_in_train_test(train_df, test_df, dataset.movies)

        dataset.movies = self.make_trend_slope_with_watches_in_time(train_df, dataset.movies)

        return train_df, test_df, test_test, dataset

    @staticmethod
    def get_movies_features(train_df, movies_df, new_movie_ids):
        mean_watch_time = train_df.groupby('movie_id')['duration'].mean()
        sum_watch_time = train_df.groupby('movie_id')['duration'].sum()
        popularity = train_df['movie_id'].value_counts()
        # movies_df.index = movies_df.id

        for movie_id in new_movie_ids.values():
            if movie_id not in mean_watch_time:
                mean_watch_time[movie_id] = 0
                sum_watch_time[movie_id] = 0
                popularity[movie_id] = 0

        return mean_watch_time, sum_watch_time, popularity, movies_df

    @staticmethod
    def get_catboost_target(test_df):
        correct_candidates_valid = test_df.groupby('user_id')['movie_id'].agg(list)

        return correct_candidates_valid
