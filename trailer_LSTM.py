from tensorflow import keras


class TrailerLSTM:
    def __init__(self, frame_num, classes_num, meta_dim):
        self.frame = frame_num
        self.classes = classes_num
        self.meta_dim = meta_dim

    def create_model(self):
        inputs = keras.layers.Input(shape=(self.frame, 1024))
        x = keras.layers.CuDNNGRU(units=256)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Flatten()(x)

        inputs_meta = keras.layers.Input(shape=self.meta_dim)
        x_meta = keras.layers.Flatten()(inputs_meta)

        x = keras.layers.concatenate([x, x_meta])
        # x = keras.layers.Dense(self.classes, activation='softmax')(x)
        x = keras.layers.Dense(1, activation='linear')(x)

        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
        model = keras.models.Model(inputs=[inputs, inputs_meta], outputs=x)

        model.compile(optimizer=opt,
                      loss='mae',
                      metrics=['mae'])

        model.summary()

        return model