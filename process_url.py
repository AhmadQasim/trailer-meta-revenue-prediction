from get_youtube_frames import download_frames
from feature_extractor.extract_features_alt import FeatureExtractor
from predict import predict
import math
import numpy as np
import os


youtube_url = input("Enter the trailer URL id: ")
DATASET_ROOT = './feature_extractor/processed_data/'
video_dir = DATASET_ROOT+'/'+youtube_url+'/'
download_frames(youtube_url, video_dir)

fe = FeatureExtractor(dataset_root=DATASET_ROOT)
fe.extract_features()

log_earnings = predict(dataset_root=DATASET_ROOT, video_id=youtube_url)


print(math.floor(math.pow(10, (np.argmax(log_earnings) * 5) / 10)))
