import os
import json
import psutil

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.dataset import Dataset
from data.preprocess import Preprocessing
from features.create_features import FeatureCreator
from models.cold_start import ColdStartRecommender
from models.first_model_training import PredModels
from models.second_model_training import CatboostModel

import gc
import warnings

warnings.filterwarnings('ignore')

from catboost import Pool


class Solution:
    def __init__(self, production_version: bool, path_to_dataset: str, users_watches_threshold: int,
                 movies_views_threshold: int, pred_models_n_candidates_few: int, pred_models_n_candidates_a_lot: int,
                n_weeks_for_catboost: int, pred_models_watches_threshold: int, n_weeks_for_local_df: int,
                catboost_train_size: float,
                ) -> None:

        self.production_version = production_version
        self.path_to_dataset = path_to_dataset

        self.users_watches_threshold = users_watches_threshold
        self.movies_views_threshold = movies_views_threshold

        self.pred_models_n_candidates_few = pred_models_n_candidates_few
        self.pred_models_n_candidates_a_lot = pred_models_n_candidates_a_lot
        self.pred_models_watches_threshold = pred_models_watches_threshold

        self.n_weeks_for_catboost = n_weeks_for_catboost
        self.n_weeks_for_local_df = n_weeks_for_local_df

        self.catboost_train_size = catboost_train_size

        self.useful_data = {}

        self.dataset = None
        self.preprocessing = None
        self.feature_creator = None
        self.pred_models = None
        self.catboost_model = None
        self.cold_start_recommender = None

        self.catboost_prediction_ids = None
        self.test_data = None
        self.both_data = None

        self.cold_start_counter = 0
        self.catboost_counter = 0

    def prepare(self):
        print('НУ ЧТО, ПОГНАЛИ!!!!')

        print('Reading data')
        self.dataset = Dataset(self.path_to_dataset)
        self.dataset.read_data()

        if self.production_version:
            self.test_data = self.dataset.logs.user_id.unique()
        print('Data read!')

        print('Preprocessing started')
        self.preprocessing = Preprocessing(users_watches_threshold=self.users_watches_threshold,
                                           movies_views_threshold=self.movies_views_threshold)
        self.dataset = self.preprocessing.preprocess(self.dataset)
        print('Preprocessing ended!')

        print('Featuring creation started!')
        self.feature_creator = FeatureCreator(
            production_version=self.production_version,
            n_weeks_for_catboost=self.n_weeks_for_catboost
        )

        self.dataset = self.feature_creator.make_features(self.dataset)

        print(f'Logs shape: {self.dataset.logs.shape}, movies shape: {self.dataset.movies.shape}')

        if not self.production_version:
            print('Пайплайн тестируется')
            d1, d2, self.test_data, self.dataset = self.feature_creator.split(self.dataset)
            self.test_data = self.test_data[self.test_data.user_id.isin(d1.user_id.unique())]
            self.test_data = self.test_data.sort_values('datetime', ascending=True)
        else:
            print('Пайплайн в проде')
            d1, d2, _, self.dataset = self.feature_creator.split(self.dataset)

        print(f'd1 size: {len(d1)}, d2 size: {len(d2)}')

        (self.useful_data['mean_watch_time'],
         self.useful_data['sum_watch_time'],
         self.useful_data['popularity'],
         self.dataset.movies) = self.feature_creator.get_movies_features(
            d1, self.dataset.movies, self.preprocessing.new_movie_ids)
        print('Featuring creation ended!')
        print('System memory:', psutil.virtual_memory())
        print('System swap memory:', psutil.swap_memory())
        print('Predmodels fitting')
        self.pred_models = PredModels(
            n_pred_candidates_few=self.pred_models_n_candidates_few,
            n_pred_candidates_a_lot=self.pred_models_n_candidates_a_lot,
            pred_models_n_threshold=self.pred_models_watches_threshold
        )

        users_watches_counter = d1.groupby('user_id')['datetime'].count()
        candidates_nn, candidates_als, candidates_cooc = self.pred_models.prepare_fit_predict(d1, users_watches_counter)

        self.useful_data['candidates_als'] = candidates_nn
        self.useful_data['candidates_nn'] = candidates_als
        self.useful_data['pred_candidates_cooc'] = candidates_cooc
        solution.dataset.predict = candidates_cooc

        catboost_target = self.feature_creator.get_catboost_target(d2)
        self.useful_data['als_item_factors'] = self.pred_models.get_als_item_factor()
        print('Predmodels fitted')

        local_logs_df = self.dataset.logs[self.dataset.logs.week_number > self.dataset.logs.week_number.max() - self.n_weeks_for_local_df].groupby('movie_id').user_id.count().sort_values(ascending=False).head(200)
        self.cold_start_recommender = ColdStartRecommender(implementation='local', local_logs_df=local_logs_df)

        del local_logs_df

        print('Catboost model created')
        self.catboost_model = CatboostModel(
            self.path_to_dataset, self.pred_models, catboost_train_size=self.catboost_train_size
        )

        self.dataset.movies[['description', 'len_genres']] = self.preprocessing.make_description_features(self.dataset.movies)
        self.dataset.movies = self.preprocessing.add_all_staff_actors_directors(self.dataset.movies)

        print('Train/test df for catboost')
        catboost_ids = d2.user_id.unique()
        print(f'Catboost ids size: {len(catboost_ids)}')

        catboost_train_df, catboost_train_groups, catboost_test_df, catboost_test_groups = (
            self.catboost_model.make_catboost_train_test_df(catboost_ids, catboost_target, candidates_als,
                                                            candidates_nn, self.useful_data['pred_candidates_cooc'],
                                                            self.dataset.movies, self.useful_data['mean_watch_time'],
                                                            self.useful_data['sum_watch_time'],
                                                            self.useful_data['popularity'], self.pred_models.items,
                                                            self.useful_data['als_item_factors']))

        print(f'Catboost train shape: {catboost_train_df.shape}, test_shape: {catboost_test_df.shape}')

        self.catboost_prediction_ids = self.dataset.logs.user_id.unique()

        print('Fitting catboost')
        self.catboost_model.fit_catboost(catboost_train_df, catboost_train_groups, catboost_test_df,
                                         catboost_test_groups)
        print('Пересчёт trend slope на всех логах')
        self.dataset.movies = self.feature_creator.make_trend_slope_with_watches_in_time(
                                                    self.dataset.logs, self.dataset.movies, window_time=10
                                                    )

        print('System memory:', psutil.virtual_memory())
        print('System swap memory:', psutil.swap_memory())

        print('Победа!')

    def pred(self, user_id):
        if user_id in self.catboost_prediction_ids and user_id not in self.catboost_model.user_id_without_predmodels:
            self.catboost_counter += 1
            catboost_predictions = self.catboost_model.catboost_predict(user_id,
                                                                        self.useful_data['candidates_als'][user_id],
                                                                        self.useful_data['candidates_nn'][user_id],
                                                                        self.useful_data['pred_candidates_cooc'],
                                                                        self.dataset.movies,
                                                                        self.useful_data['mean_watch_time'],
                                                                        self.useful_data['sum_watch_time'],
                                                                        self.useful_data['popularity'],
                                                                        self.useful_data['als_item_factors'])

            return catboost_predictions
        else:
            self.cold_start_counter += 1
            return self.cold_start_recommender.recommend(self.dataset.logs, self.dataset.movies, user_id, top_n=20)


def apk(actual, predicted, k=20):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=20):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def new_candidates_fit_models():
    sparse_matrix, top_pairs, watches_pairs = solution.pred_models.prepare_data_for_models(solution.dataset.logs)
    solution.pred_models.fit_predmodels(sparse_matrix, top_pairs)
    # candidates_nn, candidates_als, candidates_cooc = solution.pred_models.prepare_fit_predict(solution.dataset.logs)

    # solution.useful_data['candidates_nn'] = candidates_nn
    # solution.useful_data['candidates_als'] = candidates_als
    # solution.useful_data['candidates_cooc'] = candidates_cooc


def get_butch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def predict_last(BATCH_SIZE):
    print('System memory:', psutil.virtual_memory())
    print('System swap memory:', psutil.swap_memory())

    # if solution.production_version:
    users_watches_counter = solution.dataset.logs.groupby('user_id')['datetime'].count()

    for batch in tqdm(get_butch(solution.test_data, BATCH_SIZE), total=(len(solution.test_data) // BATCH_SIZE + 1)):
        current_users = []
        current_movies = []

        solution.useful_data['candidates_nn'], solution.useful_data['candidates_als'], solution.useful_data['candidates_cooc'] = \
                solution.pred_models.get_candidates(batch, [[]], users_watches_counter)

        for user_id in batch:
            current_users.append(user_id)
            current_movies.append([
                solution.preprocessing.old_movie_ids[movie_id] for movie_id in solution.pred(user_id)
            ])

            if len(current_movies[-1]) != 20:
                predict_model = 'catboost' if user_id in solution.catboost_prediction_ids and user_id not in solution.catboost_model.user_id_without_predmodels else 'coldstart'
                print(f'{user_id} предикт размера {len(current_movies[-1])}(модель: {predict_model})')

        current_predict = pd.DataFrame({
            'users': current_users,
            'movies': current_movies
        })

        predict = pd.read_csv('output/result.csv', names=['users', 'movies'])
        predict = pd.concat([predict, current_predict])
        solution.dataset.predict.to_csv('output/result.csv', index=False, header=False)

    print('Prediction saved!')


if __name__ == '__main__':
    users = []
    movies = []

    solution = Solution(
        production_version=True,
        path_to_dataset='train/',
        users_watches_threshold=0,
        movies_views_threshold=0,
        pred_models_n_candidates_few=800,
        pred_models_n_candidates_a_lot=400,
        pred_models_watches_threshold=2,
        n_weeks_for_catboost=2,
        n_weeks_for_local_df=1,
        catboost_train_size=0.999999999999,
    )

    print('System memory:', psutil.virtual_memory())
    print('System swap memory:', psutil.swap_memory())

    os.mkdir('output')
    predict = pd.DataFrame({
        'users': [],
        'movies': []
    })
    predict.to_csv('output/result.csv', index=False, header=False)

    solution.prepare()

    print('Prediction started')

    new_candidates_fit_models()
    print('System memory:', psutil.virtual_memory())
    print('System swap memory:', psutil.swap_memory())

    gc.collect()

    predict_last(BATCH_SIZE=10_000)

    print(f'Cold start counter: {solution.cold_start_counter}, catboost counter: {solution.catboost_counter}')