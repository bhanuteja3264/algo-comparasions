import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, GlobalAvgPool2D, Dense, UpSampling2D
from tensorflow.keras.models import Model

def PSPNet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    # Base features
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    
    # Pyramid pooling (reduced levels)
    pool1 = AveragePooling2D(pool_size=(8, 8))(x)
    pool2 = AveragePooling2D(pool_size=(4, 4))(x)
    
    # Merge
    x = Concatenate()([x, 
                      UpSampling2D(size=(8,8))(pool1),
                      UpSampling2D(size=(4,4))(pool2)])
    
    # Classifier
    x = GlobalAvgPool2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)