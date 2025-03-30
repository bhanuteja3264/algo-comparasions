# models/msfcn.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAvgPool2D, Dense, Multiply
from tensorflow.keras.models import Model

def MSFCN(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Lightweight encoder
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPool2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    
    # Classification head
    x = GlobalAvgPool2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)