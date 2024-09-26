import json

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from functools import partial

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from catboost import CatBoostRanker, Pool


class CatboostModel:
    def __init__(self, path_to_dataset: str, pred_models, catboost_train_size: float) -> None:
        self.path_to_dataset = path_to_dataset
        self.pred_models = pred_models
        self.catboost_train_size = catboost_train_size

        self.SIMILARITY_AGGS_MAP = {
            'similarity_mean': partial(np.mean, axis=1),
            'similarity_min': partial(np.min, axis=1),
            'similarity_max': partial(np.max, axis=1),
            'similarity_std': partial(np.std, axis=1),
            'similarity_var': partial(np.var, axis=1),
        }
        self.catboost_model = None

        self.cat_features = None
        self.features_to_drop = None
        self.text_features = None
        self.all_column_names_train = None
        self.user_id_without_predmodels = []

    @staticmethod
    def calculate_similarities(item_factors, user_candidates, user_history):
        return cosine_similarity(item_factors[user_candidates], item_factors[user_history])

    def calculate_similarity_stats(self, candidates, history):
        res = self.pred_models.model_nn.similarity[candidates] * self.pred_models.model_nn.similarity[history].T
        return res.toarray()

    @staticmethod
    def make_train_test_split(ids, train_size):
        train_ids, test_ids = train_test_split(ids, train_size=train_size, random_state=42)

        return train_ids, test_ids

    def add_features_for_catboost(self, user_id, pred_candidates_als, pred_candidates_nn,

                                  movies_df, mean_watch_time, sum_watch_time, popularity, items,
                                  als_item_factors):
        # скоры и ранки предсказанных ALS фильмов
        X1 = [item[0] for item in pred_candidates_als]
        scores1 = [item[1] for item in pred_candidates_als]
        ranks1 = [1 / i for i in range(1, len(X1) + 1)]

        # скоры и ранки предсказанных с помощью CosineRecommender фильмов
        X2 = [item[0] for item in pred_candidates_nn]
        scores2 = [item[1] for item in pred_candidates_nn]
        ranks2 = [1 / i for i in range(1, len(X2) + 1)]

        # CoOccurence
        # X3 = pred_candidates_cooc

        # Делаем датафрейм предсказанных моделями фильмов, где каждому выставляем скор и ранк
        X = np.unique(X1 + X2)
        cur_df = pl.DataFrame({
            'movie_id': pl.Series(X).cast(pl.Int64)
        })

        # Заполняем данные ALS с приведением типа ключа к i32
        als_data = pl.DataFrame({
            'movie_id': pl.Series(X1).cast(pl.Int64),
            'als_score': scores1,
            'als_rank': ranks1
        })
        nn_data = pl.DataFrame({
            'movie_id': pl.Series(X2).cast(pl.Int64),
            'nn_score': scores2,
            'nn_rank': ranks2
        })

        # Объединяем данные с основным DataFrame
        cur_df = cur_df.join(als_data, on='movie_id', how='left')
        cur_df = cur_df.join(nn_data, on='movie_id', how='left')

        # cur_df = cur_df.with_columns([
        #     pl.col("movie_id").map(lambda item_id: items_candidates_als.get(item_id, -1)).alias("similarity_to_last_als"),
        #     pl.col("movie_id").map(lambda item_id: items_candidates_nn.get(item_id, -1)).alias("similarity_to_last_nn")
        # ])
        # als_items_info = [items_candidates_als.get(item_id, -1.0) for item_id in X]
        # nn_items_info = [items_candidates_nn.get(item_id, -1.0) for item_id in X]

        # print(f'als: {als_items_info}')
        # print(f'nn: {nn_items_info}')

        # cur_df = cur_df.with_columns([
        #     pl.Series('similarity_to_last_als', als_items_info),
        #     pl.Series('similarity_to_last_nn', nn_items_info)
        # ])

        # als_items_data = pl.DataFrame({
        #     'movie_id': pl.Series(X).cast(pl.Int64),
        #     'similarity_to_last_als': pl.Series(als_items_info).cast(pl.Float64)
        # })
        #
        # nn_items_data = pl.DataFrame({
        #     'movie_id': pl.Series(X).cast(pl.Int64),
        #     'similarity_to_last_nn': pl.Series(nn_items_info).cast(pl.Float64)
        # })

        # cur_df = cur_df.join(als_items_data, on='movie_id', how='left')
        # cur_df = cur_df.join(nn_items_data, on='movie_id', how='left')

        # Обработка данных о популярности и других фичах
        popularities = pl.Series(popularity[X])
        cur_df = cur_df.with_columns([
            pl.Series('popularity', popularities),
            (pl.Series('popularity/max', popularities / popularities.max()))
        ])

        cur_videos_data = pl.from_pandas(movies_df.rename({'id': 'movie_id'}, axis=1)).filter(
            pl.col('movie_id').is_in(pl.Series(X)))

        # Добавление информации о фильмах
        movie_features = ['approx_duration', 'mean_duration', 'mean_percentage', 'rating_1', 'rating_2',
                          'rating_3', 'rating_4', 'rating_5', 'mean_score', 'median_score', 'std_score',
                          'q25_score', 'q75_score', 'trend_slope_in_7_days', 'watched_in_7_days']
        cur_df = cur_df.join(cur_videos_data.select(movie_features + ['movie_id']), on='movie_id', how='left')

        # Преобразование строки в дату
        date_reference = pl.lit(pd.Timestamp('2023-08-22'))

        # Убедитесь, что 'date_publication' в правильном формате
        cur_videos_data = cur_videos_data.with_columns(
            pl.col('date_publication').str.to_datetime(strict=False)
        )

        # Вычисление разницы в днях
        pub_date = (date_reference - cur_videos_data['date_publication']).dt.total_days()

        # Добавление нового столбца
        cur_df = cur_df.with_columns(pub_date.alias('pub_date'))

        # Обработка времени просмотра
        time_sums = pl.Series(sum_watch_time[X])
        cur_df = cur_df.with_columns([
            pl.Series('sum_time', time_sums),
            (pl.Series('sum_time/max', time_sums / time_sums.max()))
        ])

        # Среднее время просмотра
        mean_watch_times = pl.Series(mean_watch_time[X])
        cur_df = cur_df.with_columns([
            pl.Series('mean_watch_time', mean_watch_times),
            (pl.Series('mean_time/duration', mean_watch_times / (cur_df['approx_duration'] + 1)))
        ])

        # Расчет сходства
        similarities = self.calculate_similarity_stats(
            pl.Series(X).to_numpy(), pl.Series(items[user_id]).to_numpy()
        )

        # Расчет сходства для ALS
        als_similarities = self.calculate_similarities(
            als_item_factors, pl.Series(X).to_numpy(), pl.Series(items[user_id]).to_numpy()
        )

        for agg_name, agg_fn in self.SIMILARITY_AGGS_MAP.items():
            cur_df = cur_df.with_columns(
                pl.Series(agg_name, agg_fn(similarities)).alias(agg_name)
            )

        for agg_name, agg_fn in self.SIMILARITY_AGGS_MAP.items():
            cur_df = cur_df.with_columns(
                pl.Series(f'als_{agg_name}', agg_fn(als_similarities)).alias(f'als_{agg_name}')
            )

        # Нормализация скоров ALS
        if max(scores1) > 0:
            als_score_normalized = dict(zip(X1, np.array(scores1) / max(scores1)))
            cur_df = cur_df.with_columns(
                pl.col('movie_id').map_elements(lambda x: als_score_normalized.get(x, np.nan)).alias('als_score/max')
            )
        else:
            als_score_normalized = dict(zip(X1, [0] * len(X1)))
            cur_df = cur_df.with_columns(
                pl.col('movie_id').map_elements(lambda x: als_score_normalized.get(x, np.nan)).alias('als_score/max')
            )
            # Добавление меток

        # Нормализация скоров NN
        if max(scores2) > 0:
            nn_score_normalized = dict(zip(X2, np.array(scores2) / max(scores2)))
            cur_df = cur_df.with_columns(
                pl.col('movie_id').map_elements(lambda x: nn_score_normalized.get(x, np.nan)).alias('nn_score/max')
            )
        else:
            nn_score_normalized = dict(zip(X2, [0] * len(X2)))
            cur_df = cur_df.with_columns(
                pl.col('movie_id').map_elements(lambda x: nn_score_normalized.get(x, np.nan)).alias('nn_score/max')
            )

        return cur_df, X

    def make_df(self, ids, correct_candidates_test, pred_candidates_als, pred_candidates_nn, pred_candidates_cooc,
                movies_df, mean_watch_time, sum_watch_time, popularity,
                items, als_item_factors):
        groups, df = [], []

        for user_id in tqdm(ids):

            y = set(correct_candidates_test[user_id])

            # tuples (id, score)
            cur_pred_candidates_als = pred_candidates_als[user_id]
            cur_pred_candidates_nn = pred_candidates_nn[user_id]

            # только id
            als_candidates_ids = set(item[0] for item in cur_pred_candidates_als)
            nn_candidates_ids = set(item[0] for item in cur_pred_candidates_nn)
            all_pred_candidates = als_candidates_ids | nn_candidates_ids

            # сколько id правильно угадали предмодели
            true_candidates = all_pred_candidates & y

            # если предмодели не угадали ни один просмотренный им фильм, пропускаем
            if len(true_candidates) == 0:
                self.user_id_without_predmodels.append(user_id)
                continue

            false_als_candidates = als_candidates_ids - y - nn_candidates_ids
            false_nn_candidates = nn_candidates_ids - y - als_candidates_ids
            false_double_candidates = (als_candidates_ids & nn_candidates_ids) - y

            selected_false_candidates = set(np.random.choice(list(false_als_candidates),
                                                             min(len(true_candidates), len(false_als_candidates)),
                                                             replace=False)) | set(
                np.random.choice(list(false_nn_candidates),
                                 min(len(true_candidates), len(false_nn_candidates)),
                                 replace=False)) | set(np.random.choice(list(false_double_candidates),
                                                                        min(len(true_candidates),
                                                                            len(false_double_candidates)),
                                                                        replace=False))

            selected_candidates = true_candidates.union(selected_false_candidates)

            cur_pred_candidates_als = list(filter(lambda x: x[0] in selected_candidates, cur_pred_candidates_als))
            cur_pred_candidates_nn = list(filter(lambda x: x[0] in selected_candidates, cur_pred_candidates_nn))

            # self
            cur_df, X = self.add_features_for_catboost(user_id, cur_pred_candidates_als,
                                                       cur_pred_candidates_nn,
                                                       movies_df, mean_watch_time, sum_watch_time, popularity, items,
                                                       als_item_factors)

            labels = [int(item in y) for item in X]
            cur_df = cur_df.with_columns(pl.Series('label', labels))

            groups += [user_id] * len(cur_df)
            df.append(cur_df)

        df = pl.concat(df)
        df = df.to_pandas().fillna(0).reset_index(drop=True)

        return df, groups

    def make_catboost_train_test_df(self, ids, correct_candidates_test,
                                    pred_candidates_als, pred_candidates_nn, pred_candidates_cooc,
                                    movies_df, mean_watch_time, sum_watch_time, popularity, items_df,
                                    als_item_factors):
        train_ids, test_ids = self.make_train_test_split(ids, self.catboost_train_size)

        final_train_df, train_groups = self.make_df(train_ids, correct_candidates_test, pred_candidates_als,
                                                    pred_candidates_nn, pred_candidates_cooc, movies_df,
                                                    mean_watch_time,
                                                    sum_watch_time, popularity, items_df, als_item_factors)

        final_test_df, test_groups = self.make_df(test_ids, correct_candidates_test, pred_candidates_als,
                                                  pred_candidates_nn, pred_candidates_cooc,
                                                  movies_df, mean_watch_time, sum_watch_time, popularity, items_df,
                                                  als_item_factors)

        # all_staff = movies_df['staff'].apply(lambda x: set(x))
        # all_actors = movies_df['actor'].fillna('[]').apply(lambda x: set(x))
        # all_directors = movies_df['director'].fillna('[]').apply(lambda x: set(x))

        # for movie_id in movies_df.id:
        #     movies_df.loc[movie_id, 'staff'] = str(
        #         list(all_staff[movie_id] - (all_actors[movie_id] | all_directors[movie_id])))

        final_train_df = final_train_df.merge(movies_df.rename({'id': 'movie_id'}, axis=1)[
                                                  ['movie_id', 'description', 'staff', 'name', 'len_genres', 'director',
                                                   'actor']],
                                              on='movie_id', how='left').rename(columns={'description': 'text'})
        final_test_df = final_test_df.merge(movies_df.rename({'id': 'movie_id'}, axis=1)[
                                                ['movie_id', 'description', 'staff', 'name', 'len_genres', 'director',
                                                 'actor']],
                                            on='movie_id', how='left').rename(columns={'description': 'text'})

        final_train_df = final_train_df.fillna('')
        final_test_df = final_test_df.fillna('')

        final_train_df = final_train_df.rename(columns={'description': 'text'})
        final_test_df = final_test_df.rename(columns={'description': 'text'})

        return final_train_df, train_groups, final_test_df, test_groups

    def fit_catboost(self, train_df, train_groups, test_df, test_groups, save=False):
        # print(train_df.columns)
        self.cat_features = []
        self.features_to_drop = ['label', 'movie_id']
        self.text_features = ['text', 'staff', 'name', 'director', 'actor']
        self.all_column_names_train = train_df.drop(self.features_to_drop, axis=1).columns.to_list()

        train_pool = Pool(
            data=train_df.drop(self.features_to_drop, axis=1)[self.all_column_names_train],
            label=train_df['label'],
            group_id=train_groups,
            cat_features=self.cat_features,
            text_features=self.text_features
        )

        # test_pool = Pool(
        #     data=test_df.drop(self.features_to_drop, axis=1),
        #     label=test_df['label'],
        #     group_id=test_groups,
        #     cat_features=self.cat_features,
        #     text_features=self.text_features
        # )

        params = {
            'task_type': 'CPU',
            'loss_function': 'YetiRank',
            # 'custom_metric': ['MAP:top=20', 'MAP:top=10', 'MAP:top=5', 'AUC', 'F1'],
            'iterations': 500,
        }

        self.catboost_model = CatBoostRanker(**params, random_seed=42)
        # self.catboost_model.fit(train_pool, eval_set=test_pool, verbose=True, use_best_model=True)
        self.catboost_model.fit(train_pool, verbose=True, use_best_model=True)

        if save:
            self.catboost_model.save_model(f'{self.path_to_dataset}catboost_model.cbm')

    def catboost_predict(self, user_id, candidates_als, candidates_nn, candidates_cooc,
                         movies_df, mean_watch_time, sum_watch_time, popularity, als_item_factors):
        # candidates_nn = self.pred_models.predict_user_with_implicit(self.pred_models.model_nn, user_id,
        #                                               self.pred_models.items[user_id])
        # candidates_als = self.pred_models.predict_user_with_implicit(self.pred_models.model_als, user_id,
        #                                                self.pred_models.items[user_id])

        cur_df, X = self.add_features_for_catboost(user_id, candidates_als,
                                                   candidates_nn,
                                                   movies_df,
                                                   mean_watch_time,
                                                   sum_watch_time,
                                                   popularity,
                                                   self.pred_models.items,
                                                   als_item_factors)
        
        # all_staff = movies_df['staff'].apply(lambda x: set(x))
        # all_actors = movies_df['actor'].fillna('[]').apply(lambda x: set(x))
        # all_directors = movies_df['director'].fillna('[]').apply(lambda x: set(x))

        # for movie_id in movies_df.id:
        #     movies_df.loc[movie_id, 'staff'] = str(
        #         list(all_staff[movie_id] - (all_actors[movie_id] | all_directors[movie_id])))

        # movies_df['director'] = movies_df['director'].fillna('').apply(list).apply(lambda x: ' '.join(map(str, x)))
        # movies_df['actor'] = movies_df['actor'].fillna('').apply(list).apply(lambda x: ' '.join(map(str, x)))

        cur_df = cur_df.to_pandas().fillna(0).reset_index(drop=True)
        cur_df = cur_df.merge(movies_df.rename({'id': 'movie_id'}, axis=1)[
                                  ['movie_id', 'description', 'staff', 'name', 'len_genres', 'director', 'actor']],
                              on='movie_id', how='left').rename(columns={'description': 'text'})
        
        cur_df['text'] = cur_df['text'].fillna('')
        cur_df['staff'] = cur_df['staff'].fillna('')

        groups = [user_id] * len(X)

        test_pool = Pool(
            data=cur_df[self.all_column_names_train],
            group_id=groups,
            cat_features=self.cat_features,
            text_features=self.text_features
        )

        try:
            preds = self.catboost_model.predict(test_pool)
            idx = range(len(X))
            idx = sorted(idx, key=lambda x: preds[x], reverse=True)
            candidates = list(X[idx])
        except Exception:
            print(user_id)
            candidates = range(21)
        return candidates[:20]
