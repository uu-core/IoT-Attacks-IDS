from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, auc, average_precision_score, confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import logging
import os
import torch


# ===========================
# Updated Evaluation Metrics Function
# ===========================
def evaluate_metrics(y_true, y_pred, y_prob, test_domain_name, training_domain_name):
    """
    Compute evaluation metrics including confusion matrix, specificity, and balanced accuracy,
    plot the ROC curve, log and print the results, and return all metrics in a dictionary.
    
    Args:
      y_true (np.array): True labels.
      y_pred (np.array): Predicted labels.
      y_prob (np.array): Predicted probabilities for the positive class.
      test_domain_name (str): Name of the test domain.
      training_domain_name (str): Name of the training domain.
      
    Returns:
      metrics (dict): Dictionary with evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Compute confusion matrix and additional metrics
    cm = confusion_matrix(y_true, y_pred)
    # logging.info(f"Confusion matrix shape: {cm.shape}, values: \n{cm}")
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    log_msg = (f"Train Domain: {training_domain_name} | Test Domain: {test_domain_name} | "
               f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | "
               f"F1: {f1:.4f} | "
               f"Specificity: {specificity:.4f} | Balanced Acc: {balanced_acc:.4f} | "
               f"CM: {cm.tolist()}")    # ROC-AUC: {roc_auc:.4f} 
    # print(log_msg)
    # logging.info(log_msg)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm
    }

