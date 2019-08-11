
import json
import pandas as pd
import pickle
import os
import fire
import shutil

DATASET_ROOT = './feature_extractor/youtube_data/'
JSON_FILE = './crawled_revenue.json'
TMDB_FILE = './tmdb_data/meta_features.csv'


f = open(JSON_FILE, "r")
titles = json.load(f)

meta_features = pd.read_csv(TMDB_FILE)
print(meta_features.head())

titles_keys = list(titles.keys())


def save_meta_features():
    for idx, tconst in enumerate(titles_keys):
        title = titles[tconst]
        if title['YoutubeURL'] is not '':
            if title['YoutubeURL'] in os.listdir(DATASET_ROOT):
                entry = meta_features.loc[meta_features['imdb_id'] == tconst]
                entry = entry.drop(columns=['imdb_id'])
                path = DATASET_ROOT + title['YoutubeURL'] + '/X2'
                with open(path, 'wb') as filepath:
                    pickle.dump(entry, filepath)
                print(idx, tconst)


def print_meta_features():
    sum = 0
    for idx, tconst in enumerate(titles_keys):
        title = titles[tconst]
        if title['YoutubeURL'] is not '':
            if title['YoutubeURL'] in os.listdir(DATASET_ROOT):
                path = DATASET_ROOT + title['YoutubeURL'] + '/X2'
                path_no_x2 = DATASET_ROOT + title['YoutubeURL']
                with open(path, 'rb') as filepath:
                    try:
                        x = pickle.load(filepath)
                        if x.empty:
                            sum += 1
                            shutil.rmtree(path_no_x2)
                    except Exception as e:
                        print("Exception: ", e, " ", path)
                        sum += 1
                        shutil.rmtree(path_no_x2)
    print("Empty: ", sum)


def load_crawled_budgets():
    sum = 0
    crawl_file = open('crawled_budget_revenue.txt', 'r')
    data = json.load(crawl_file)
    data_keys = list(data.keys())
    for key in data_keys:
        entry = data[key]
        if entry['budget'] != " N/A\n":
            sum += 1
    print("Budget available for: ", sum)


if __name__ == '__main__':
    fire.Fire()