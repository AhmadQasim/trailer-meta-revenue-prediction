from data_generator import DataGenerator
import os
from tensorflow import keras
from trailer_LSTM import TrailerLSTM
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import ShuffleSplit

DATASET_ROOT = './feature_extractor/youtube_data/'
BATCH_SIZE = 15
FRAMES = 32
EPOCHS_PER_FOLD = 10
TEST_RATIO = 0.1
# total epochs = FOLDS * EPOCHS_PER_FOLDS
FOLDS = 10
DIM = (360, 480)
CHANNELS = 3
NUM_CLASSES = 20

lstm = TrailerLSTM(FRAMES, NUM_CLASSES)
model = lstm.create_model()

mcp_save = keras.callbacks.ModelCheckpoint('./models/model_lstm_lr_categorical.h5', save_best_only=True,
                                           monitor='val_mae', mode='min')

video_ids = np.array(os.listdir(DATASET_ROOT))
test_indices = list(ShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=1).split(video_ids))

video_ids_test = video_ids[test_indices[0][1]]
video_ids = video_ids[test_indices[0][0]]

folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=1).split(video_ids))

for i, indices in enumerate(folds):

    print("Fold : ", i)

    video_ids_train = video_ids[indices[0]]
    video_ids_val = video_ids[indices[1]]

    # testing the data generator
    train_dg = DataGenerator(video_ids_train, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS,
                             n_classes=NUM_CLASSES)
    valid_dg = DataGenerator(video_ids_val, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS,
                             n_classes=NUM_CLASSES)

    history = model.fit_generator(generator=train_dg,
                                  validation_data=valid_dg,
                                  steps_per_epoch=len(video_ids_train)/BATCH_SIZE,
                                  epochs=EPOCHS_PER_FOLD, callbacks=[mcp_save])


test_dg = DataGenerator(video_ids_test, BATCH_SIZE, frames=FRAMES, dim=DIM, n_channels=CHANNELS, n_classes=NUM_CLASSES,
                        dataset_root=DATASET_ROOT)

result = model.evaluate_generator(test_dg)
print(result)