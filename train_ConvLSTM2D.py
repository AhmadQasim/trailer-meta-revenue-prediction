from data_generator_frames import DataGenerator
import os
from tensorflow import keras

DATASET_ROOT = './feature_extractor/youtube_data/'
BATCH_SIZE = 12
TRAIN_RATIO = 0.1
FRAMES = 8
EPOCHS = 50
DIM = (360, 480)
CHANNELS = 3

# read the list of all video ids
video_ids = os.listdir(DATASET_ROOT)
total_size = len(video_ids)

train_size = int(total_size*TRAIN_RATIO)
video_ids_train = video_ids[:train_size]

# testing the data generator
train_dg = DataGenerator(video_ids_train, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS)

inputs = keras.layers.Input(shape=(FRAMES, *DIM, CHANNELS))
x = keras.layers.ConvLSTM2D(filters=1, kernel_size=DIM, data_format='channels_last')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1, activation="linear")(x)

opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
model = keras.models.Model(inputs=inputs, outputs=x)

model.compile(optimizer=opt,
              loss='mean_absolute_percentage_error')

model.summary()

history = model.fit_generator(generator=train_dg,
                              validation_data=train_dg,
                              workers=6,
                              epochs=EPOCHS)
