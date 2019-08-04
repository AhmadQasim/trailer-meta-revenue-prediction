from data_generator import DataGenerator
import os
from tensorflow import keras
from trailer_LSTM import TrailerLSTM

DATASET_ROOT = './feature_extractor/youtube_data/'
BATCH_SIZE = 15
TRAIN_RATIO = 0.9
FRAMES = 32
EPOCHS = 100
DIM = (360, 480)
CHANNELS = 3
NUM_CLASSES = 20

# read the list of all video ids
video_ids = os.listdir(DATASET_ROOT)
total_size = len(video_ids)

train_size = int(total_size * TRAIN_RATIO)
valid_size = total_size - train_size
video_ids_train = video_ids[:train_size]
video_ids_valid = video_ids[train_size:train_size + valid_size]

# testing the data generator
train_dg = DataGenerator(video_ids_train, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS,
                         n_classes=NUM_CLASSES)
valid_dg = DataGenerator(video_ids_valid, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS,
                         n_classes=NUM_CLASSES)

lstm = TrailerLSTM(FRAMES, NUM_CLASSES)
model = lstm.create_model()

mcp_save = keras.callbacks.ModelCheckpoint('./models/model_lstm_lr_categorical.h5', save_best_only=True,
                                           monitor='val_acc', mode='max')

history = model.fit_generator(generator=train_dg,
                              validation_data=valid_dg,
                              epochs=EPOCHS, callbacks=[mcp_save])
