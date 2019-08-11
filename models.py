import numpy as np
from tensorflow import keras
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def cross_validation(model, X_train, y_train, splits=5):
    model = LinearRegression(fit_intercept=False)
    scores = cross_val_score(model, X_train, y_train, cv=splits)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def neural_net(X_train, x_val, y_train, y_val, epochs=50):
    dims = X_train.shape[1]
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_dim=dims, activation='relu'))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation="linear"))

    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(optimizer=opt,
                  loss='mae',
                  metrics=['mae'])

    model.summary()
    history = model.fit(X_train, np.array(y_train), epochs=epochs, batch_size=64,
                        validation_data=(x_val, np.array(y_val)),
                        verbose=1)
    return model


def linear_regression(X_train, y_train, rigde=False, regularization=0.3):
    """
    :param X_train: training data features
    :param y_train: training data labels
    :param rigde: boolean to specify whether to do linear or ridge regression
    :param regularization: ridge-regression strength
    :return: regression model
    """
    if rigde:
        model = Ridge(regularization, fit_intercept=False)
    else:
        model = LinearRegression(fit_intercept=False)

    model.fit(X_train, y_train)
    cross_validation(model, X_train, y_train)
    return model


def support_vector_regression(X_train, y_train):
    model = SVR(gamma='scale', C=1.0, epsilon=0.2)
    model.fit(X_train, y_train)
    return model


def random_forest_regression(X_train, y_train):
    model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    model.fit(X_train, y_train)
    return model


def pca_regression(X_train, y_train):
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_train)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_pca, y_train)
    return model


# Variable Selection with RFECV
def recursive_feature_elimination(X_train, y_train, estimator='lr'):
    if estimator == 'lr':
        model = linear_regression(X_train, y_train)
    if estimator == 'pcar':
        model = pca_regression(X_train, y_train)
    if estimator == 'rf':
        model = random_forest_regression(X_train, y_train)
    if estimator == 'svr':
        model = support_vector_regression(X_train, y_train)

    rfecv = RFECV(estimator=estimator, step=1, cv=KFold(n_splits=5), scoring='r2')
    model = rfecv.fit(X_train, y_train)
    X_reduced = model.transform(X_train)
    print("Useful features: ", rfecv.n_features_)
    return model
