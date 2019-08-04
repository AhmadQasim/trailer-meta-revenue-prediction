import numpy as np
from tensorflow import keras
import os
import cv2
import pickle


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    '''
    arguments

    list_IDs: list of video_ids
    dim: dimensions of the image (height, width)
    dataset_root: the root folder of dataset
    frames: number of frames to be considered
    '''

    def __init__(self, list_IDs, batch_size=4, dim=(360, 480), n_channels=3,
                 shuffle=True, dataset_root='./feature_extractor/youtube_data/', frames=64):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dataset_root = dataset_root
        self.frames = frames
        self.on_epoch_end()

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
        X = np.empty((self.batch_size, self.frames, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, video_id in enumerate(list_IDs_temp):
            video_dir = self.dataset_root+video_id+'/'
            frames_root = video_dir+'frames/'
            # list the frames saved within the corresponding directory
            for frame, im in enumerate(os.listdir(frames_root)):
                img = cv2.imread(frames_root + im)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X[i, frame, :, :] = img

                if frame == self.frames-1:
                    break

            # Store budget
            with open(self.dataset_root + video_id + '/y', "rb") as f:
                y[i] = pickle.load(f)
        # X = np.expand_dims(X, axis=1)
        return X, y
