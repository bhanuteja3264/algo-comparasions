import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves confusion matrix
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for saving the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'results/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

def plot_sample_predictions(model, test_gen, num_samples=3):
    """
    Plots sample predictions vs ground truth
    Args:
        model: Trained model
        test_gen: Test generator
        num_samples: Number of samples to plot
    """
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, (x, y) in enumerate(test_gen):
        if i >= num_samples:
            break
            
        pred = model.predict(x)
        pred = (pred > 0.5).astype(np.float32)
        
        plt.subplot(num_samples, 3, 3*i+1)
        plt.imshow(x[0])
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i+2)
        plt.imshow(y[0].squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i+3)
        plt.imshow(pred[0].squeeze(), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.savefig('results/sample_predictions.png', bbox_inches='tight')
    plt.close()