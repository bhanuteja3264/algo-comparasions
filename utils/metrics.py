import tensorflow as tf
import numpy as np

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for binary segmentation
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Dice coefficient
    """
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (Jaccard index) for binary segmentation
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor
    Returns:
        IoU score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def specificity(y_true, y_pred):
    """
    Specificity (True Negative Rate)
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
    Returns:
        Specificity score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    tn = tf.keras.backend.sum((1 - y_true_f) * (1 - y_pred_f))
    fp = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    return tn / (tn + fp + tf.keras.backend.epsilon())

def get_metrics():
    """Returns list of metrics for model compilation"""
    return [dice_coef, iou_score, 'accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]