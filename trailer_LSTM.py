from tensorflow import keras


class TrailerLSTM:
    def __init__(self, frame_num, classes_num):
        self.frame = frame_num
        self.classes = classes_num

    def create_model(self):
        inputs = keras.layers.Input(shape=(self.frame, 1024))
        x = keras.layers.CuDNNGRU(units=256)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(self.classes, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='linear')(x)

        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
        model = keras.models.Model(inputs=inputs, outputs=x)

        model.compile(optimizer=opt,
                      loss='mae',
                      metrics=['mae'])

        model.summary()

        return model
