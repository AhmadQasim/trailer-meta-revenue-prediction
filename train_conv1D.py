from data_generator import DataGenerator
import os
from tensorflow import keras
from trailer_conv1D import TrailerConv1D

DATASET_ROOT = './feature_extractor/youtube_data/'
BATCH_SIZE = 12
TRAIN_RATIO = 0.9
FRAMES = 64
EPOCHS = 50
NUM_CLASSES = 9

# read the list of all video ids
video_ids = os.listdir(DATASET_ROOT)
total_size = len(video_ids)

train_size = int(total_size*TRAIN_RATIO)
valid_size = total_size - train_size
video_ids_train = video_ids[:train_size]
video_ids_valid = video_ids[train_size:train_size+valid_size]

# testing the data generator
train_dg = DataGenerator(video_ids_train, BATCH_SIZE, frames=FRAMES, n_classes=NUM_CLASSES)
valid_dg = DataGenerator(video_ids_valid, BATCH_SIZE, frames=FRAMES, n_classes=NUM_CLASSES)

conv1d = TrailerConv1D(FRAMES, NUM_CLASSES)
model = conv1d.create_model()

mcp_save = keras.callbacks.ModelCheckpoint('./models/model_conv1D.h5', save_best_only=True,
                                           monitor='val_acc', mode='max')

history = model.fit_generator(generator=train_dg,
                              validation_data=valid_dg,
                              workers=6,
                              epochs=EPOCHS,
                              callbacks=[mcp_save])
