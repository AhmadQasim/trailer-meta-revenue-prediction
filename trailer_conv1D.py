from tensorflow import keras


class TrailerConv1D:
    def __init__(self, frame_num, classes_num):
        self.frame = frame_num
        self.classes = classes_num

    def create_model(self):
        inputs = keras.layers.Input(shape=(self.frame, 1024))
        # input to this layer: (BATCH_SIZE, 64, 1024) with channel as the last
        # activation set to relu and strides set to 2
        x = keras.layers.Conv1D(input_shape=(self.frame, 1024), filters=1024, kernel_size=8,
                                activation='relu', strides=2, data_format='channels_last')(inputs)
        # output of last layer: (BATCH_SIZE, 29, 1024)
        conv1d_out_normalized = keras.layers.BatchNormalization()(x)
        # input to this layer: Batch Normalized (BATCH_SIZE, 29, 1024)
        x = keras.layers.Conv1D(filters=1024, kernel_size=1, activation='relu')(conv1d_out_normalized)
        x = keras.layers.BatchNormalization()(x)
        # perform element wise addition between x with shape (BATCH_SIZE, 29, 1024) and conv1d_out with shape
        # (BATCH_SIZE, 29, 1024)
        x = keras.layers.Add()([x, conv1d_out_normalized])
        # output of last layer: Batch Normalized (BATCH_SIZE, 29, 1024) which is the input to average pooling layer
        # apply relu before propagating through the averagepooling1d layer
        x = keras.layers.ReLU()(x)
        # x = keras.layers.AveragePooling1D(pool_size=29)(x)
        x = keras.layers.Flatten()(x)
        # output of last layer: (BATCH_SIZE, 1024)
        x = keras.layers.Dense(self.classes, activation="softmax")(x)
        # output of last layer: (BATCH_SIZE, 1)

        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
        model = keras.models.Model(inputs=inputs, outputs=x)

        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        return model
