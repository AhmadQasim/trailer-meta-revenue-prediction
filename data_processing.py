# ref: https://medium.com/@nagalr_63588/tmdb-box-office-prediction-kaggle-com-6e14e013955b

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ast

def read_crawled_data(file_name, is_json=False):
    # read crawled data file
    f = open(file_name, "r")
    if is_json:
        return json.load(f)
    crawled_data = []
    for line in f:
        crawled_data.append(line)
    return crawled_data

def normalize_json(data):
    res = []
    for id, rev in data.items():
        res.append({'imdb_id': id, 'revenue': rev})
    return res


def parse_json(x):
    try:
        res = json.loads(x.replace("'", '"'))
        print("HERE: ", res, type(res))
        return json.loads(x.replace("'", '"'))[0]['name']
    except:
        return ''


def text_to_list(x):
    if pd.isna(x):
        return ''
    else:
        if "[" not in x:
            return x
        else:
            return ast.literal_eval(x)


def parse_genre(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['genres1', 'genres2', 'genres3'])
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['genres1', 'genres2', 'genres3'])
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['genres1', 'genres2', 'genres3'])
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['genres1', 'genres2', 'genres3'])


def parse_production_companies(x):
    if type(x) == str:
        return pd.Series(['','',''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) == 1:
        return pd.Series([x[0]['name'],'',''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],''], index=['prod1', 'prod2', 'prod3'] )
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['prod1', 'prod2', 'prod3'] )


def parse_release_date(df):
    df['release_date'] = pd.to_datetime(df['release_date'], format='%m/%d/%y')
    df['weekday'] = df['release_date'].dt.weekday
    df['weekday'].fillna(df['weekday'].median(), inplace=True)
    df['month'] = df['release_date'].dt.month
    df['month'].fillna(df['month'].median(), inplace=True)
    df['year'] = df['release_date'].dt.year
    df['year'].fillna(df['year'].median(), inplace=True)
    df['day'] = df['release_date'].dt.day
    df['day'].fillna(df['day'].median(), inplace=True)
    df.drop(columns=['release_date'], inplace = True)
    return df


def parse_cast(x):
    cast = ['cast1', 'cast2', 'cast3', 'cast4', 'cast5']
    out = [-1]*5
    if type(x) != str:
        for i in range(min([5,len(x)])):
            out[i] = x[i]['id']
    return pd.Series(out, index=cast)


def parse_crew(x):
    crew = ['Director', 'Producer']
    out = [-1] * 2
    if type(x) != str:
        for item in x:
            if item['job'] == 'Director':
                out[0] = item['id']
            elif item['job'] == 'Producer':
                out[1] = item['id']
    return pd.Series(out, index=crew)


# TODO: method to find if given entry has budget information, return boolean
def has_budget(data):
    return True


def transform_categorical_to_numerical(df, cols):
    items = list(set(df[cols].values.ravel().tolist()))
    encoder = LabelEncoder()
    encoder.fit(items)
    df[cols] = df[cols].apply(lambda x: encoder.transform(x))
    return df


def join_to_features():
    # merge tmdb_train and tmdb_test datasets
    train_data = pd.read_csv('tmdb_data/train.csv')
    train_data = train_data.drop('revenue', axis=1)
    test_data = pd.read_csv('tmdb_data/test.csv')
    tmdb_data = pd.concat((train_data, test_data), sort=False)

    # read crawled revenues
    rev = read_crawled_data('crawled_revenue.txt', is_json=True)
    norm_rev = normalize_json(rev)
    rev_df = pd.DataFrame.from_dict(norm_rev, orient='columns')
    combined = pd.merge(tmdb_data, rev_df, on='imdb_id')

    # prepare data
    combined.drop(columns=['id',
                           'poster_path',
                           'title',
                           'original_title',
                           'Keywords',
                           'spoken_languages',
                           'belongs_to_collection',
                           'homepage',
                           'tagline',
                           'production_countries',
                           'overview',
                           'status',
                           'popularity'], inplace=True)
    for col in ['genres', 'production_companies', 'cast', 'crew']:
        combined[col] = combined[col].apply(text_to_list)

    # parse genres
    combined[['genres1', 'genres2', 'genres3']] = combined['genres'].apply(parse_genre)
    combined.drop(columns=['genres'], inplace=True)

    # parse productions companies
    combined[['prod1', 'prod2', 'prod3']] = combined['production_companies'].apply(parse_production_companies)
    combined.drop(columns=['production_companies'], inplace=True)

    # format release date
    combined = parse_release_date(combined)

    # parse cast
    combined[['cast1', 'cast2', 'cast3', 'cast4', 'cast5']] = combined['cast'].apply(parse_cast)
    combined.drop(columns=['cast'], inplace=True)

    # parse crew
    combined[['director', 'producer']] = combined['crew'].apply(parse_crew)
    combined.drop(columns=['crew'], inplace=True)

    # fill missing with most common values
    combined['runtime'].fillna(combined['runtime'].median(), inplace = True)
    combined['original_language'].fillna('en', inplace=True)
    combined.drop(columns='imdb_id', inplace=True)

    # create log-budget
    combined['budget_log'] = np.log1p(combined['budget'])

    # convert categorical features to numerical values
    cols = ['genres1', 'genres2', 'genres3']
    combined = transform_categorical_to_numerical(combined, cols)
    cols = ['prod1', 'prod2', 'prod3']
    combined = transform_categorical_to_numerical(combined, cols)
    cols = ['original_language']
    combined = transform_categorical_to_numerical(combined, cols)

    return combined


def main():
    join_to_features()


if __name__ == '__main__':
    main()
