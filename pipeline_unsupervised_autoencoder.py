import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from table_merge import *


def add_autoencoder(data):
    input_layer = Input(shape=(data.shape[1],))
    encoded = Dense(336, activation='relu')(input_layer)
    encoded = Dense(112, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(data.shape[1], activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(data, data, epochs=50, batch_size=256,
                    shuffle=True, validation_split=0.2, verbose=1)

    reconstructed_data = autoencoder.predict(data)
    return pd.DataFrame(reconstructed_data, columns=data.columns), autoencoder

def autoencoder_classifier(X_train, X_test, y_train, y_test, task_name, random_state):

    # entfernen von allen Zeilen mit dem wert True in y_Train und X_Train
    mask = ~y_train
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Hinzufügen des Autoencoders und Rückgabe des autoencoder
    X_train_autoencoded, autoencoder = add_autoencoder(X_train)

    # Anomalieerkennung auf dem Testdatensatz
    X_test_autoencoded = autoencoder.predict(X_test)

    # Berechnen der Fehlerquadrate (Mean Squared Error, MSE)
    mse = np.mean(np.square(X_test - X_test_autoencoded), axis=1)

    y_pred = mse > 0.0646

    name = autoencoder_classifier.__name__

    return y_test, y_pred, name, task_name, random_state







'''# Berechnen des Rekonstruktionsfehlers für Testdaten
reconstruction_error_test = np.mean(np.square(X_test.values - X_test_autoencoded.values))
print("Reconstruction Error on Test Data:", reconstruction_error_test)

mae = np.mean(np.abs(X_train.values - X_train_autoencoded.values))
print("Mean Absolute Error on Training Data:", mae)

kl_divergence = np.mean(kullback_leibler_divergence(X_train.values, X_train_autoencoded.values))
print("KL-Divergence on Training Data:", kl_divergence)'''
