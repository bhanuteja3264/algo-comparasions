import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, GlobalAveragePooling2D, Dense, UpSampling2D
from tensorflow.keras.models import Model

def DeeplabV3(input_shape=(128, 128, 3)):
    """Lightweight DeepLabV3 implementation for binary classification"""
    inputs = Input(shape=input_shape)
    
    # Base features
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    
    # ASPP Lite (all branches maintain spatial dimensions)
    pool = AveragePooling2D(pool_size=(1, 1))(x)  # Using (1,1) to maintain shape
    conv1 = Conv2D(32, 1, activation='relu')(x)
    conv2 = Conv2D(32, 3, dilation_rate=6, activation='relu', padding='same')(x)
    
    # Resize pool branch to match others (identity operation)
    pool = UpSampling2D(size=(1, 1))(pool)
    
    # Merge features
    x = Concatenate()([pool, conv1, conv2])
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='DeepLabV3')