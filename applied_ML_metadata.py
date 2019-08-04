"""
    Creates model and tests model accuracy
    usage (in Edit Configurations->Parameters): -method='nn' -save=False -vis=False
"""
import pandas as pd
import urllib.request
import json
import fire
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from models import linear_regression, \
    support_vector_regression, \
    random_forest_regression,\
    pca_regression,\
    neural_net,\
    recursive_feature_elimination
from data_processing import join_to_features

print(tf.__version__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# TODO: K-fold cross validation
# TODO: Join new budget info to table
# TODO: Create df and subsequent models without budget information

def read_data(titles_file, ratings_file=None):
    titles = pd.read_csv(titles_file, delimiter='\t')
    ratings = pd.read_csv(ratings_file, delimiter='\t') if ratings_file else None
    return titles, ratings


def find_movies_opening_week_earnings(titles):
    titles = titles[titles.startYear != "\\N"]
    titles["startYear"] = pd.to_numeric(titles["startYear"])
    titles_filtered = titles[(titles.titleType == 'movie') & (titles.startYear < 1990) & (titles.startYear > 1970)]
    print(len(titles_filtered))
    f = open("omdb_tmp_21052019.dump", "a")

    cnt = 0
    for index, title in titles_filtered.iterrows():
        try:
            contents = urllib.request.urlopen("http://www.omdbapi.com/?i={}&apikey=2e92a86a".format(title['tconst'])).read()
            f.write(contents.decode("utf-8") + ",\n")
        except Exception as e:
            print("Error: ", e)
            continue
        cnt += 1
        if cnt > 100000:
            break


def load_and_save_json_omdb(filename):
    f = open(filename, "r")
    titles = json.load(f)
    titles_boxoffice = []
    for title in titles:
        if "BoxOffice" in title:
            if title["BoxOffice"] != "N/A":
                titles_boxoffice.append(title)
    return titles_boxoffice


def data_vectorization(data):
    sentences = []
    y = []
    for entry in data:
        actors = format_actor_names(entry)
        sentences.append(entry["Title"] + " " +
                         entry["Released"] + " " +
                         entry["Country"] + " " +
                         entry["Language"] + " " +
                         entry["Production"].replace(' ', '') + " " +
                         entry["Director"].replace(' ', '') + " " +
                         entry["Writer"].replace(' ', '') + " " +
                         entry["Genre"].replace(',', ' ') + " " +
                         actors
                         )
        y.append(get_revenue(entry))

    vectorizer = CountVectorizer(lowercase=True)
    vectorizer.fit(sentences)
    vocab_size = len(vectorizer.vocabulary_)
    X = vectorizer.transform(sentences).todense()
    test_sentence = (X[0], y[0])
    return X, y, vocab_size, test_sentence


def format_actor_names(names):
    # remove spaces between actor first and last names, make full name single feature
    actors = ""
    actor_tokens = names["Actors"].split(",")
    for token in actor_tokens:
        actors = actors + " " + token.replace(' ', '')
    return actors


def get_revenue(data):
    # remove comma and dollar sign from revenue, convert shorthands (k, M) accordingly if present
    revenue_str = data["BoxOffice"].replace(',', '').replace('$', '')
    if 'k' in revenue_str:
        revenue = float(revenue_str.replace('k', '')) * 1000.0
    elif 'M' in revenue_str:
        revenue = float(revenue_str.replace('M', '')) * 1000000.0
    else:
        revenue = float(revenue_str)
    return revenue


def get_train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=50)
    return X_train, X_test, x_val, y_train, y_test, y_val


def save_model(name, model):
    with open(name, 'wb') as picklefile:
        pickle.dump(model, picklefile)


def load_model(name):
    with open(name, 'rb') as training_model:
        model = pickle.load(training_model)
    return model


def test_model(model, X_test, y_test, is_pca=False):
    if is_pca:
        pca = PCA(n_components=8)
        X_test = pca.fit_transform(X_test)
    # make predictions using the test set
    y_pred = model.predict(X_test)

    # mean squared error
    print("Mean abs error: %.2f"
          % mean_absolute_error(y_test, y_pred))
    # explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    return y_pred


def visualize_accuracy(y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()

# MEAN ABS ERRORS
# Linear Regression (vectorized features): 288,673,070.73
# Neural Net (vectorized features): 20,249,589.57
# Linear Regression w/ budget: 7,286,245.26
# Random Forest w/ budget: 7,776,798.64
# Support Vector Regression w/ budget: 10,306,380.77
# PCA Regression w/ budget: 13,157,483.85
# Neural Net w/budget: 7,589,397.16


def main(method='lr', save=False, vis=False):
    '''
    pd.options.mode.chained_assignment = None
    titles, ratings = read_data('imdb_data/title.basics.tsv', 'imdb_data/title.ratings.tsv')
    '''

    # pre-processing data
    X = join_to_features()
    y = X['revenue']
    X.drop(columns='revenue', inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=50)

    is_pca = False
    is_nn = False

    if method == 'lr':
        model = linear_regression(X_train, y_train)
    if method == 'pcar':
        model = pca_regression(X_train, y_train)
        is_pca = True
    if method == 'rf':
        model = random_forest_regression(X_train, y_train)
    if method == 'svr':
        model = support_vector_regression(X_train, y_train)
    if method == 'nn':
        model = neural_net(X_train, x_val, y_train, y_val, 2)
        is_nn = True

    predictions = test_model(model, X_test, y_test, is_pca=is_pca)

    if save:
        if is_nn:
            model.save('models/nn_model.h5')
        else:
            save_model('models/'+method+'_model', model)

    if vis:
        visualize_accuracy(y_test, predictions)


if __name__ == "__main__":
    fire.Fire(main)
