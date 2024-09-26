import numpy as np
import pandas as pd
import implicit
from scipy import sparse

from typing import List, Tuple, Dict
from collections import Counter, defaultdict

from tqdm import tqdm


class CoocurenceRecommender:
    def __init__(
            self,
            top_pairs: Dict[int, List[Tuple[int, int]]],
            item_index: int = 0,
            num_recs: int = 300,
    ):
        self.top_pairs = top_pairs
        self.item_index = item_index
        self.num_recs = num_recs

    @staticmethod
    def get_seqential_user_ids(recs, user_ids) -> np.ndarray:
        return np.repeat(user_ids, [len(r) for r in recs])

    @staticmethod
    def get_weights():
        return pd.read_csv('src/models/cooc_weights/weights.csv', names=['movie', 'weights'])
    
    @staticmethod
    def save_weights():
        return pd.to_csv('src/models/cooc_weights/weights.csv')

    def recommend(self, data: List[List[int]], user_ids) -> pd.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in data
        ]

        predict = pd.DataFrame({
            'user_id': self.get_seqential_user_ids(recs, user_ids),
            'item_id': [item[0] for rec in recs for item in rec],
            f'cooc_score_{self.item_index}': [item[1] for rec in recs for item in rec],
            f'cooc_rank_{self.item_index}': [rnk for rec in recs for rnk, _ in enumerate(rec)],
        })

        return predict

    def _recommend_one_user(self, user_data: List[int]):
        if len(user_data) <= self.item_index:
            return []

        # top_pair_items = self.top_pairs.get(user_data[-self.item_index - 1], [(2480, 0), (484, 0), (6194, 0), (641, 0), (5849, 0), (1281, 0)])
        top_pair_items = self.top_pairs.get(user_data[-self.item_index - 1], [(886, 0), (170, 0), (2264, 0), (222, 0), (2110, 0), (444, 0)])

        us = set(user_data)
        res = [item for item in top_pair_items if item[0] not in us][:self.num_recs]

        return res


class PredModels:
    def __init__(self, n_pred_candidates_few: int, n_pred_candidates_a_lot: int, pred_models_n_threshold: int):
        self.n_pred_candidates_few = n_pred_candidates_few
        self.n_pred_candidates_a_lot = n_pred_candidates_a_lot

        self.pred_models_n_threshold = pred_models_n_threshold

        self.items = None
        self.ids = None
        self.model_nn = None
        self.model_als = None
        self.cooc_models = None

    @staticmethod
    def make_sparse_matrix(train_df):
        enum_items = np.array(train_df["movie_id"].values).astype(int)
        enum_users = np.array(train_df["user_id"].values)

        sparse_matrix = sparse.csr_matrix(
            (np.ones(shape=(len(enum_users))), (enum_users, enum_items)),
        )

        return sparse_matrix
    
    @staticmethod
    def get_top_pairs_from_data(data: List[List[int]], max_candidates: int = 500):
        pairs = defaultdict(Counter)
        top_pairs = {}

        for record in data:
            for i in range(len(record) - 1):
                last, ans = record[i], record[i + 1]
                pairs[last][ans] += 1

        for key in pairs:
            top_pairs[key] = pairs[key].most_common(max_candidates)

        return top_pairs
    
    @staticmethod
    def outer_merge(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame) -> pd.DataFrame:
        common_columns = list(set(first_dataframe.columns) & set(second_dataframe.columns))
        return pd.merge(first_dataframe, second_dataframe, on=common_columns, how='outer')

    def set_items_and_ids(self, train_df):
        self.items = train_df.groupby('user_id')['movie_id'].agg(list)
        self.ids = train_df.user_id.unique()
    
    def prepare_data_for_models(self, train_df):
        sparse_matrix = self.make_sparse_matrix(train_df)

        self.set_items_and_ids(train_df)

        watches_pairs = [[]]
        top_pairs = {}
        # watches_pairs = train_df.groupby('user_id')['movie_id'].apply(list).to_list()
        # top_pairs = self.get_top_pairs_from_data(watches_pairs)

        return sparse_matrix, top_pairs, watches_pairs
    
    def predict_user_with_implicit(self, model, user_id, user_items, user_watches_count):
        enum_users = np.zeros(len(user_items)).astype(int)
        enum_items = np.array(list(user_items)).astype(int)

        cur_sparse_matrix = sparse.csr_matrix(
            (np.ones(shape=(len(enum_users))), (enum_users, enum_items))
        )

        if user_watches_count[user_id] < self.pred_models_n_threshold:
            N = self.n_pred_candidates_few
        else:
            N = self.n_pred_candidates_a_lot

        rec = model.recommend(user_id, cur_sparse_matrix, N=N, recalculate_user=False,
                        filter_already_liked_items=True)

        idx = range(len(rec[0]))
        idx = sorted(idx, key=lambda x: rec[1][x], reverse=True)
        return list(zip(rec[0][idx], rec[1][idx]))
    
    # @staticmethod
    # def predict_item_similarity(model, item_id):
    #     rec = model.similar_items(item_id, N=800)
    #     idx = range(len(rec[0]))
    #     idx = sorted(idx, key=lambda x: rec[1][x], reverse=True)
    #     return list(zip(rec[0][idx], rec[1][idx]))

    def fit_predmodels(self, sparse_matrix, top_pairs):
        self.model_nn = implicit.nearest_neighbours.BM25Recommender(210, 1.1, 0.4)
        self.model_nn.fit(sparse_matrix)

        self.model_als = implicit.als.AlternatingLeastSquares(
                            factors=20, alpha=13, random_state=42
                        )
        print("start fitting als")
        self.model_als.fit(sparse_matrix)
        print("finished fitting als")

        self.cooc_models = [
            CoocurenceRecommender(top_pairs, 0, 300)
        ]

    def get_candidates(self, ids_to_get, watches_pairs, user_watches_count):
        predicted_candidates_nn = {}
        predicted_candidates_als = {}
        predicted_candidates_cooc = {}

        for user_id in ids_to_get:
            if user_id in self.ids:
                candidates_nn = self.predict_user_with_implicit(self.model_nn, user_id, self.items[user_id], user_watches_count)
                candidates_als = self.predict_user_with_implicit(self.model_als, user_id, self.items[user_id], user_watches_count)
            else:
                candidates_nn = []
                candidates_als = []

            predicted_candidates_nn[user_id] = candidates_nn
            predicted_candidates_als[user_id] = candidates_als

        # cooc_df = self.cooc_models[0].recommend(watches_pairs, ids_to_get)
        # for model in self.cooc_models[1:]:
        #     first_df = model.recommend(watches_pairs, ids_to_get)
        #     cooc_df = self.outer_merge(cooc_df, first_df)
        
        predicted_candidates_cooc = self.cooc_models[0].get_weights()

        return predicted_candidates_nn, predicted_candidates_als, predicted_candidates_cooc

    # def get_items_candidates(self):
    #     pred_candidates_nn_items = {}
    #     pred_candidates_als_items = {}
    #
    #     for user in self.ids:
    #         candidates = self.predict_item_similarity(
    #             self.model_nn, self.items[user][0]
    #         )
    #         pred_candidates_nn_items[user] = dict(candidates)
    #
    #     for user in self.ids:
    #         candidates = self.predict_item_similarity(
    #             self.model_als, self.items[user][0]
    #         )
    #         pred_candidates_als_items[user] = dict(candidates)
    #
    #     return pred_candidates_nn_items, pred_candidates_als_items

    def prepare_fit_predict(self, train_df, user_watches_count):
        sparse_matrix, top_pairs, watches_pairs = self.prepare_data_for_models(train_df)
        self.fit_predmodels(sparse_matrix, top_pairs)
        print("fitting cooc")
        cooc_pred_ids = train_df.groupby('user_id')['movie_id'].apply(list).index.to_list()
        print("get candidates")
        candidates_nn, candidates_als, candidates_cooc = self.get_candidates(self.ids, watches_pairs, user_watches_count)
        print("finished")

        return candidates_nn, candidates_als, candidates_cooc
    
    def get_als_item_factor(self):
        return self.model_als.item_factors
