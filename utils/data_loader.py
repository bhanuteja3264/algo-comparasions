import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(config):
    """
    Creates train, validation, and test generators with augmentation
    Args:
        config (dict): Configuration dictionary with:
            - train_dir (str): Path to training data
            - test_dir (str): Path to test data
            - input_shape (tuple): Target image size (h, w, c)
            - batch_size (int): Batch size
            - val_split (float): Validation split ratio
    Returns:
        Three generators: train_gen, val_gen, test_gen
    """
    # Augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=config['val_split']
    )

    # No augmentation for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    train_gen = train_datagen.flow_from_directory(
        config['train_dir'],
        target_size=config['input_shape'][:2],
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='training',
        shuffle=True,
        color_mode='rgb' if config['input_shape'][2] == 3 else 'grayscale'
    )

    # Validation generator
    val_gen = train_datagen.flow_from_directory(
        config['train_dir'],
        target_size=config['input_shape'][:2],
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='validation',
        shuffle=False,
        color_mode='rgb' if config['input_shape'][2] == 3 else 'grayscale'
    )

    # Test generator
    test_gen = test_datagen.flow_from_directory(
        config['test_dir'],
        target_size=config['input_shape'][:2],
        batch_size=1,  # Important for accurate evaluation
        class_mode='binary',
        shuffle=False,
        color_mode='rgb' if config['input_shape'][2] == 3 else 'grayscale'
    )

    print(f"\nFound {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    print(f"Found {test_gen.samples} test images\n")

    return train_gen, val_gen, test_gen