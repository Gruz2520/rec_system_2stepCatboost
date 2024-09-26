import gc
import pandas as pd


class Dataset:
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.countries = None
        self.genres = None
        self.movies = None
        self.logs = None
        self.staff = None
        self.predict = None

    def read_data(self):
        self.countries = pd.read_csv(f'{self.path_to_dataset}countries.csv')
        self.genres = pd.read_csv(f'{self.path_to_dataset}genres.csv')
        self.movies = pd.read_csv(f'{self.path_to_dataset}movies.csv')
        self.logs = pd.read_csv(f'{self.path_to_dataset}logs.csv')
        self.staff = pd.read_csv(f'{self.path_to_dataset}staff.csv')

    def save_data(self):
        self.countries.to_csv(f'{self.path_to_dataset}countries_processed.csv', index=False)
        self.genres.to_csv(f'{self.path_to_dataset}genres_processed.csv', index=False)
        self.movies.to_csv(f'{self.path_to_dataset}movies_processed.csv', index=False)
        self.logs.to_csv(f'{self.path_to_dataset}logs_processed.csv', index=False)
        self.staff.to_csv(f'{self.path_to_dataset}staff_processed.csv', index=False)
