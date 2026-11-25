import keras
from keras import layers


def mlp(x, hidden_units, dropout_rate):
    """Stack of Dense (GELU) + Dropout layers."""
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
