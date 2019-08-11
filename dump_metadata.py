import os
import json
import pandas as pd

DATASET_ROOT = './feature_extractor/youtube_data/'
JSON_FILE = './crawler_revenue.json'
TMDB_FILE = './tmdb_data/meta_features.csv'


f = open(JSON_FILE, "r")
titles = json.load(f)

meta_features = pd.read_csv(TMDB_FILE)

titles_keys = list(titles.keys())

for idx, tconst in enumerate(titles_keys):
    print(idx, tconst)
    title = titles[tconst]
    if title['YoutubeURL'] is not '':
