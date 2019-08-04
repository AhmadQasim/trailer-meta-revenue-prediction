from data_generator import DataGenerator
import os
from tensorflow import keras
import numpy as np

DATASET_ROOT = './feature_extractor/youtube_data/'
BATCH_SIZE = 1
TEST_RATIO = 0.2
FRAMES = 32
EPOCHS = 300
DIM = (360, 480)
CHANNELS = 3
MODEL = 'model_lstm_lr_categorical.h5'
NUM_CLASSES = 20

# read the list of all video ids
video_ids = os.listdir(DATASET_ROOT)
total_size = len(video_ids)

test_size = int(total_size*TEST_RATIO)
video_ids_test = video_ids[:test_size]

test_dg = DataGenerator(video_ids_test, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS, n_classes=NUM_CLASSES,
                        dataset_root=DATASET_ROOT)

model = keras.models.load_model('./models/' + MODEL)

result = model.evaluate_generator(test_dg)
print(result)

for idx, item in enumerate(test_dg):
    earnings = model.predict(item[0], batch_size=1)
    print(np.argmax(item[1]), np.argmax(earnings))
    if idx == 20:
        break
