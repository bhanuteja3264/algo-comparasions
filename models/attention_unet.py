# models/attention_unet.py
import tensorflow as tf
from tensorflow.keras.layers import Multiply

def AttentionUNet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPool2D()(x)
    
    # Attention gate
    g = Conv2D(32, 1, activation='relu')(x)
    x = Multiply()([x, g])
    
    # Classifier
    x = GlobalAvgPool2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)