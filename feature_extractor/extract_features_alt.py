import os
from feature_extractor import feature_extractor
import numpy as np
import cv2
import pickle
import tensorflow as tf

# script to extract the feature vectors of each video. The feature vectors are 
# saved into the corresponding folder with the name x (x has the dimension
# (1024,120). If the amount of key frames in a video is smaller than 120, the 
# remaining vectors of x get filled with 0)


class FeatureExtractor:
    def __init__(self, dataset_root='./feature_extractor/processed_data/'):
        self.DATASET_ROOT = dataset_root
        self.extractor = feature_extractor.YouTube8MFeatureExtractor()
        self.frame_count = 64
        self.video_ids = os.listdir(self.DATASET_ROOT)

    def extract_features(self):
        for i, video_id in enumerate(self.video_ids):
            features = np.empty((self.frame_count, 1024), np.dtype('float32'))
            video_dir = self.DATASET_ROOT+video_id+'/'
            frames_root = video_dir+'frames/'
            # list the frames saved within the corresponding directory
            for frame, im in enumerate(os.listdir(frames_root)):
                img = cv2.imread(frames_root + im)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.extractor.extract_rgb_frame_features(img)
                features[frame, :] = img
            with open(video_dir+'X', "wb") as f:
                pickle.dump(features, f)
            print('All feature vectors of directory: ' + video_dir + ' have been extracted ' + str(i))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
fe = FeatureExtractor()
fe.extract_features()
