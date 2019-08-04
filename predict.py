from tensorflow import keras
import pickle
import numpy as np


def predict(dataset_root, video_id):
    FRAMES = 32
    MODEL = 'model_lstm_lr_categorical.h5'

    X = load_data(dataset_root, video_id, FRAMES)
    model = keras.models.load_model('./models/' + MODEL)

    earnings = model.predict(X, batch_size=1)
    return earnings


def load_data(dataset_root, video_id, frames):
    # Initialization
    X = np.zeros((1, frames, 1024))

    with open(dataset_root + video_id + '/X', "rb") as f:
        X_vid = pickle.load(f)
        X[0, :, :] = X_vid[:frames, :]

    return X
