import json
import os

import ast
import numpy as np
import pandas as pd


class Preprocessing:
    def __init__(self, movies_views_threshold, users_watches_threshold):
        self.movies_views_threshold = movies_views_threshold
        self.users_watches_threshold = users_watches_threshold

        self.new_movie_ids = None
        self.old_movie_ids = None

        self.staff_columns_names = ['actor', 'producer', 'editor', 'director',
                                    'writer', 'composer', 'presenter', 'commentator']

    @staticmethod
    def day_part(hour):
        time_of_day = {
            (0,1,2,3,4,5): "night",
            (6,7,8,9,10,11): "morning",
            (12,13,14,15,16,17): "day",
            (18,19,20,21,22,23): "evening"
        }

        for hours, part in time_of_day.items():
            if hour in hours:
                return part

    @staticmethod
    def add_columns_to_df(df, columns):
        for column in columns:
            df[column] = np.nan
            df[column] = df[column].astype(object)

        return df

    @staticmethod
    def set_columns_types(df, columns, type_to_set):
        df[columns] = df[columns].astype(type_to_set)

        return df

    def set_new_old_ids(self, movies_df, logs_df):
        self.new_movie_ids = { movies_df.id.iloc[i]: i for i in range(movies_df.shape[0]) }
        self.old_movie_ids = { self.new_movie_ids[i]: i for i in self.new_movie_ids }

        movies_df = movies_df.reset_index(drop=True)
        movies_df.id = movies_df.id.apply(lambda x: self.new_movie_ids[x])
        logs_df.movie_id = logs_df.movie_id.apply(lambda x: self.new_movie_ids[x])

        return movies_df, logs_df

    def make_datetime_features(self, logs_df):
        logs_df['day'] = logs_df['datetime'].dt.day
        logs_df['day_of_week'] = logs_df['datetime'].dt.day_of_week
        logs_df['day_of_year'] = logs_df['datetime'].dt.day_of_year
        logs_df['hour'] = logs_df['datetime'].dt.hour
        logs_df['day_part'] = logs_df['hour'].apply(self.day_part)
        logs_df['is_weekend'] = np.where(logs_df['day_of_week'].isin([5,6]), 1,0)
        logs_df['week_number'] = logs_df['datetime'].dt.isocalendar().week

        return logs_df
    
    @staticmethod
    def count_views_for_movie(logs_df, result_length):
        views_count = logs_df.groupby('movie_id')['user_id'].nunique()
        views_count = views_count.reindex(range(result_length), fill_value=0)

        return views_count
    
    @staticmethod
    def get_year_of_release(movies_df):
        return pd.to_datetime(movies_df['year']).dt.year
    
    @staticmethod
    def filter_movies(movies_df, movies_views_threshold):
        # movies_df = movies_df[~((movies_df.genres == '[]') & (movies_df.staff == '[]'))]
        # movies_df = movies_df[movies_df.year <= 2023]
        movies_df = movies_df[movies_df.views > movies_views_threshold]

        return movies_df
    
    @staticmethod
    def clear_logs_from_missing_movies(logs_df, movies_df):
        logs_df = logs_df[logs_df.movie_id.isin(movies_df.id)]

        return logs_df


    @staticmethod
    def group_aggregate_logs(logs_df):
        return logs_df.groupby(['user_id', 'movie_id'], as_index=False, sort=False).agg({
                'datetime': 'first',
                'duration': 'sum',
                'day': 'first',
                'day_of_week': 'first',
                'day_of_year': 'first',
                'hour': 'first',
                'day_part': 'first',
                'is_weekend': 'first',
                'week_number': 'first'
            }).reset_index(drop=True)
    
    @staticmethod
    def add_countries_genres_data_to_logs(logs_df, movies_df):
        ids = movies_df.id.unique().tolist()

        for i in ids:
            ids_in_logs = logs_df[logs_df.movie_id == i].index.tolist()
            movie_info = movies_df.loc[movies_df.id == i]
            for j in ['genres', 'countries']:
                value = movie_info[j].values[0][1:-1].split(',')

                if value != ['']:
                    for k in ids_in_logs:
                        logs_df.at[k, j] = value

        return logs_df

    @staticmethod
    def split_staff_by_role_movies(movies_df, staff_df, staff_columns):
        ids = movies_df.id.unique().tolist()
        for i in ids:
        # список сделан строкой, поэтому надо его разбить и обработать
            list_of_staff = json.loads(movies_df.staff.loc[i])

            if len(list_of_staff):
                df_for_movie = staff_df.loc[staff_df.id.isin(list_of_staff)]
                
                for j in staff_columns:
                    movies_df.at[i, j] = df_for_movie[df_for_movie.role == j].id.to_list()

        return movies_df

    @staticmethod
    def filter_logs_by_users_watches(logs_df, users_watches_threshold):
        return logs_df.groupby('user_id').filter(lambda x: len(x) > users_watches_threshold)

    def preprocess(self, dataset):
        dataset.logs['datetime'] = pd.to_datetime(dataset.logs['datetime'], format='mixed')
        dataset.logs = self.make_datetime_features(dataset.logs)

        dataset.logs = self.set_columns_types(dataset.logs, ['user_id', 'movie_id'], int)

        dataset.movies['views'] = self.count_views_for_movie(dataset.logs, dataset.movies.shape[0])
        dataset.movies['year'] = self.get_year_of_release(dataset.movies)

        dataset.movies = self.filter_movies(dataset.movies, self.movies_views_threshold)
        dataset.logs = self.clear_logs_from_missing_movies(dataset.logs, dataset.movies)

        dataset.logs = dataset.logs.sort_values('datetime')
        dataset.logs = self.group_aggregate_logs(dataset.logs)

        dataset.logs = self.add_columns_to_df(dataset.logs, ['genres', 'countries'])
        dataset.logs = self.add_countries_genres_data_to_logs(dataset.logs, dataset.movies)

        dataset.movies = self.add_columns_to_df(dataset.movies, self.staff_columns_names)
        dataset.movies = self.split_staff_by_role_movies(dataset.movies, dataset.staff, self.staff_columns_names)

        dataset.movies, dataset.logs = self.set_new_old_ids(dataset.movies, dataset.logs)

        dataset.logs = self.filter_logs_by_users_watches(dataset.logs, self.users_watches_threshold)

        return dataset
    
    @staticmethod
    def prep(text, bad_set):
        for c in bad_set:
            if c in text:
                text = text.replace(c, '')
        return text

    def make_description_features(self, movies_df):
            local_df = movies_df[['id','name', 'description', 'genres']]
            genres = local_df.genres.to_list()
            new_gen = []

            for i in range(len(genres)):
                new_gen.append(eval(genres[i]))
                
            ids_sorted = pd.Series(pd.Series(new_gen).sum()).value_counts().index.to_list()
            sorted_genres = []

            for i in range(len(new_gen)):
                local_cnt = 0
                local_str = ''
                for j in range(len(ids_sorted)):
                    if ids_sorted[j] in new_gen[i]:
                        local_cnt += 1 
                        local_str += str(ids_sorted[j]) + '.'
                    if local_cnt==3 or local_cnt==len(new_gen[i]):
                        sorted_genres.append(local_str[:-1])
                        break

            local_df['sorted_genres'] = sorted_genres
            local_df['len_genres'] = [len(i) for i in new_gen]
            local_df['description'] = local_df.description.apply(str).apply(lambda s: s.lower())
            bad_set = set(local_df.description.sum())
            bad_set2 = ''

            for c in bad_set:
                if c not in ' йцукенгшщзхъфывапролджэячсмитьбю':
                    bad_set2 += c

            local_df['description'] = local_df.description.apply(lambda x: self.prep(x, bad_set2))

            return local_df[['description', 'len_genres']]
    
    @staticmethod
    def add_all_staff_actors_directors(movies_df):
        all_staff = movies_df['staff'].apply(lambda x: set(x))
        all_actors = movies_df['actor'].fillna('[]').apply(lambda x: set(x))
        all_directors = movies_df['director'].fillna('[]').apply(lambda x: set(x))

        for movie_id in movies_df.id:
            movies_df.loc[movie_id, 'staff'] = str(list(all_staff[movie_id] - (all_actors[movie_id] | all_directors[movie_id])))

        movies_df['staff'] = movies_df['staff'].apply(lambda x: x.replace(',', '').replace('[', '').replace(']', ''))

        movies_df['director'] = movies_df['director'].fillna('').apply(list).apply(lambda x: ' '.join(map(str, x)))
        movies_df['actor'] = movies_df['actor'].fillna('').apply(list).apply(lambda x: ' '.join(map(str, x)))
        return movies_df

if __name__ == '__main__':
    pass
    # print('Starting...')
    # preproccesing = Preprocessing('Data/')
    #
    # preproccesing.read_data()
    # print('Data readed')
    #
    # print('Preprocessing...')
    # preproccesing.preprocess()
    # print('Preprocessing finished')
    #
    # preproccesing.save_data()
    # print('Data saved')