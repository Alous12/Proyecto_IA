import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from src import config


def construir_cnn(tasa_aprendizaje=0.001):
    model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, 3, activation='relu', strides=(2, 2), padding='same'),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(2),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=tasa_aprendizaje), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def entrenar(X, y, X_val, y_val, epocas=20, tam_lote=128, peso_real=1.0, tasa_aprendizaje=0.001,
             paciencia=5, semilla=config.SEMILLA, verbose=True):
    tf.keras.utils.set_random_seed(semilla)
    model = construir_cnn(tasa_aprendizaje)
    parada = EarlyStopping(monitor='val_loss', patience=paciencia, restore_best_weights=True)
    history = model.fit(X.reshape(-1, 64, 64, 3), y, validation_data=(X_val.reshape(-1, 64, 64, 3), y_val),
                        epochs=epocas, batch_size=tam_lote, class_weight={0: peso_real, 1: 1.0},
                        callbacks=[parada], verbose=2 if verbose else 0)
    perdidas = history.history
    historial = {"epoca": list(range(1, len(perdidas["loss"]) + 1)),
                 "error_train": perdidas["loss"], "error_val": perdidas["val_loss"]}
    return model, historial, int(np.argmin(perdidas["val_loss"])) + 1


def predecir_probabilidad(X, model):
    return model.predict(X.reshape(-1, 64, 64, 3), batch_size=256, verbose=0).ravel()
