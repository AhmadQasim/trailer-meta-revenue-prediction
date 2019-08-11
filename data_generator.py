import numpy as np
from tensorflow import keras
import pickle
from feature_extractor.feature_extractor import YouTube8MFeatureExtractor
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    '''
    arguments
    
    list_IDs: list of video_ids
    dim: dimensions of the image (height, width)
    dataset_root: the root folder of dataset
    frames: number of frames to be considered
    '''
    def __init__(self, list_IDs, batch_size=4, dim=(480, 360), n_channels=3,
                 shuffle=True, dataset_root='./feature_extractor/youtube_data/', frames=64,
                 n_classes=9):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dataset_root = dataset_root
        self.frames = frames
        self.on_epoch_end()
        self.extractor = YouTube8MFeatureExtractor()
        self.featureLength = 1024
        self.n_classes = n_classes

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        # shuffle the video ids after every epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.zeros((self.batch_size, self.frames, self.featureLength))
        # y = np.zeros((self.batch_size, self.n_classes), dtype=int)
        y = np.zeros(self.batch_size, dtype=int)

        X2 = np.zeros((self.batch_size, 1, 22))

        # Generate data
        for i, video_id in enumerate(list_IDs_temp):
            # list the frames saved within the corresponding directory
            with open(self.dataset_root + video_id + '/X', "rb") as f:
                X_vid = pickle.load(f)
                X[i, :, :] = X_vid[:self.frames, :]

            with open(self.dataset_root + video_id + '/X2', "rb") as f:
                X2_vid = pickle.load(f)
                X2_vid = X2_vid.to_numpy()
                X2[i, 0, :] = X2_vid

            # Store budget
            with open(self.dataset_root+video_id+'/y', "rb") as f:
                val = pickle.load(f)
                # val = 5 * round(math.floor(math.log10(val) * 10) / 5)
                # y[i, int(val/5)] = 1
                # val = math.floor(math.log10(val) * 100)
                y[i] = val
        # X = np.expand_dims(X, axis=1)
        return [X, X2], y
