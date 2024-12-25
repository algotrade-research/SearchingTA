from src import Searching
from utils import *

downloader = Downloader()

data = downloader.get_historical_data(start_date='2022-01-01', end_date='2023-02-01')

search = Searching(data=data, TP=(1, 5), SL=(1, 5), number_of_trials=10, side='long', mode='one_way', n_jobs=1)

if __name__ == "__main__":
    search.run()