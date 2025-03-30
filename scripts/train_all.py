import sys
from pathlib import Path
import os
import yaml
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import models
from models.msfcn import MSFCN
from models.deeplabv3 import DeeplabV3
from models.segnet import SegNet
from models.pspnet import PSPNet
from models.unet import UNet
from utils.data_loader import create_data_generators

def verify_model_shapes(model, input_shape):
    """Verify model can process sample input"""
    try:
        test_input = tf.random.normal([1] + list(input_shape))
        _ = model(test_input)
        return True
    except Exception as e:
        print(f"Shape verification failed: {str(e)}")
        return False

def train_model(model, name, train_gen, val_gen, config):
    if not verify_model_shapes(model, config['input_shape']):
        print(f"Skipping {name} due to shape mismatch")
        return None
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            f"models/{name}_best.keras",
            save_best_only=True
        )
    ]
    
    print(f"\n=== Training {name} ===")
    history = model.fit(
        train_gen,
        steps_per_epoch=min(150, len(train_gen)),
        validation_data=val_gen,
        validation_steps=min(30, len(val_gen)),
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

if __name__ == '__main__':
    # Load config
    with open("configs/params.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create data generators
    train_gen, val_gen, _ = create_data_generators(config)
    
    # Initialize models with verified input shapes
    models = {
        "MSFCN": MSFCN(config['input_shape']),
        "DeepLabV3": DeeplabV3(config['input_shape']),
        "SegNet": SegNet(config['input_shape']),
        "PSPNet": PSPNet(config['input_shape']),
        "UNet": UNet(config['input_shape'])
    }
    
    # Train models
    results = {}
    for name, model in models.items():
        try:
            history = train_model(model, name, train_gen, val_gen, config)
            if history:
                results[name] = {
                    'accuracy': max(history.history['val_accuracy']),
                    'auc': max(history.history['val_auc'])
                }
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = f"results/training_results_{timestamp}.csv"
    
    import pandas as pd
    pd.DataFrame(results).T.to_csv(results_file)
    print(f"\nResults saved to {results_file}")