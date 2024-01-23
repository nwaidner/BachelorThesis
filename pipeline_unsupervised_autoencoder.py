import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from table_merge import *


def add_autoencoder(data):
    input_layer = Input(shape=(data.shape[1],))
    encoded = Dense(112, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(112, activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='relu')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(data, data, epochs=50, batch_size=256,
                    shuffle=True, validation_split=0.2, verbose=1)

    reconstructed_data = autoencoder.predict(data)
    return pd.DataFrame(reconstructed_data, columns=data.columns), autoencoder

def autoencoder(X_train, X_test, y_train, y_test, task_name, random_state, threshold):
    mask = ~y_train
    X_train = X_train[mask]
    y_train = y_train[mask]

    X_train_autoencoded, autoencoder = add_autoencoder(X_train)

    X_test_autoencoded = autoencoder.predict(X_test)

    mse = np.mean(np.square(X_test - X_test_autoencoded), axis=1)

    y_pred = mse > 0.1

    name = autoencoder.__name__

    return y_test, y_pred, name, task_name, random_state

