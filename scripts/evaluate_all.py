import sys
from pathlib import Path
import os
import yaml
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import data loader
from utils.data_loader import create_data_generators

def plot_confusion_matrix(cm, model_name):
    """Save confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['No Tumor', 'Tumor'])
    plt.yticks([0, 1], ['No Tumor', 'Tumor'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white' if cm[i][j] > cm.max()/2 else 'black')
    
    os.makedirs("results/confusion_matrices", exist_ok=True)
    plt.savefig(f"results/confusion_matrices/{model_name}_cm.png")
    plt.close()

def evaluate_model(model_path, test_gen, model_name):
    """Evaluate a single model"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Get predictions
        y_pred_probs = model.predict(test_gen)
        y_pred = (y_pred_probs > 0.5).astype(int)
        y_true = test_gen.classes
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, target_names=['No Tumor', 'Tumor'], output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        
        # Save visualization
        plot_confusion_matrix(cm, model_name)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['Tumor']['precision'],
            'recall': report['Tumor']['recall'],
            'f1_score': report['Tumor']['f1-score'],
            'roc_auc': roc_auc,
            'specificity': report['No Tumor']['recall'],  # TN / (TN + FP)
            'confusion_matrix': cm.tolist()
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

if __name__ == '__main__':
    # Load config
    with open("configs/params.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create test data generator
    _, _, test_gen = create_data_generators(config)
    
    # List of all models to evaluate (including Attention UNet)
    model_names = [
        'MSFCN',
        'UNet',
        'AttentionUNet',  # Added Attention UNet
        'DeepLabV3',
        'SegNet',
        'PSPNet'
    ]
    
    results = {}
    
    # Evaluate each model
    for name in model_names:
        model_path = f"models/{name}_best.keras"
        if os.path.exists(model_path):
            print(f"\nEvaluating {name}...")
            metrics = evaluate_model(model_path, test_gen, name)
            if metrics:
                results[name] = metrics
                print(f"{name} evaluation complete!")
                print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        else:
            print(f"Model not found: {model_path}")
    
    # Save comprehensive results
    os.makedirs("results", exist_ok=True)
    results_file = "results/evaluation_results.csv"
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results).T
    df.to_csv(results_file, float_format='%.4f')
    
    print("\n=== Evaluation Summary ===")
    print(df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(4))
    print(f"\nDetailed results saved to {results_file}")
    print("Confusion matrices saved to results/confusion_matrices/")