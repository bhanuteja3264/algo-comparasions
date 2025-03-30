import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, GlobalAvgPool2D, Dense
from tensorflow.keras.models import Model

def SegNet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPool2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    
    # Decoder (simplified)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    
    # Classifier
    x = GlobalAvgPool2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)