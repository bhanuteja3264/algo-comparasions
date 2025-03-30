# models/unet.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, GlobalAvgPool2D, Dense
from tensorflow.keras.models import Model

def UNet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPool2D()(c1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    
    # Decoder
    u1 = UpSampling2D()(c2)
    x = Concatenate()([c1, u1])
    
    # Classifier
    x = GlobalAvgPool2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)