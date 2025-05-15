import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, recall_score

'''
Script of metrics : Mean  ( IoU, F1, recall, precision), per class (IoU, F1, Recall, Precision)
'''

def iou(y_true, y_pred, epsilon=1e-6):
    """Compute Intersection over Union for two binary masks."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)

def f1_score(y_true, y_pred, epsilon=1e-6):
    """Compute the F1 score (Dice coefficient) for two binary masks."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    return (2 * intersection + epsilon) / (y_true.sum() + y_pred.sum() + epsilon)

def recall(y_true, y_pred, epsilon=1e-6):
    """Compute recall for two binary masks."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    TP = (y_true * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()
    return TP / (TP + FN + epsilon)

def compute_metrics(eval_pred, num_classes=5, ignore_index=0, epsilon=1e-6):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    N, H_lab, W_lab = labels.shape
    _, H_pred, W_pred = preds.shape

    if (H_pred, W_pred) != (H_lab, W_lab):
        scale_h = H_lab // H_pred
        scale_w = W_lab // W_pred
        preds = np.repeat(np.repeat(preds, scale_h, axis=1), scale_w, axis=2)

    y_true = labels.reshape(-1)
    y_pred = preds.reshape(-1)

    iou_per_class      = {}
    f1_per_class       = {}
    recall_per_class   = {}
    precision_per_class = {}

    for c in range(num_classes):
        true_c = (y_true == c).astype(np.uint8)
        pred_c = (y_pred == c).astype(np.uint8)

        tp = np.logical_and(true_c, pred_c).sum()
        fp = np.logical_and(1 - true_c, pred_c).sum()
        fn = np.logical_and(true_c, 1 - pred_c).sum()

        union = tp + fp + fn

        iou       = (tp + epsilon) / (union + epsilon)
        f1        = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        recall    = (tp + epsilon) / (tp + fn + epsilon)
        precision = (tp + epsilon) / (tp + fp + epsilon)

        iou_per_class[f"iou_class_{c}"]         = iou
        f1_per_class[f"f1_class_{c}"]           = f1
        recall_per_class[f"recall_class_{c}"]   = recall
        precision_per_class[f"precision_class_{c}"] = precision

    valid = [c for c in range(num_classes) if c != ignore_index]
    mean_iou       = np.mean([iou_per_class[f"iou_class_{c}"]         for c in valid])
    mean_f1        = np.mean([f1_per_class[f"f1_class_{c}"]           for c in valid])
    mean_recall    = np.mean([recall_per_class[f"recall_class_{c}"]   for c in valid])
    mean_precision = np.mean([precision_per_class[f"precision_class_{c}"] for c in valid])

    metrics = {
        "mean_iou_no_bg":       mean_iou,
        "mean_f1_no_bg":        mean_f1,
        "mean_recall_no_bg":    mean_recall,
        "mean_precision_no_bg": mean_precision,
    }
    metrics.update(iou_per_class)
    metrics.update(f1_per_class)
    metrics.update(recall_per_class)
    metrics.update(precision_per_class)

    return metrics

def compute_metrics_per_class(p):
    """
    p.predictions: logits de forma [N, C, H, W]
    p.label_ids:   etiquetas [N, H, W]
    """
    preds = np.argmax(p.predictions, axis=1).reshape(-1)
    labels = p.label_ids.reshape(-1)
    num_classes = p.predictions.shape[1]

    ious = []
    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.logical_and(preds == c, labels == c).sum()
        fp = np.logical_and(preds == c, labels != c).sum()
        fn = np.logical_and(preds != c, labels == c).sum()
        union = tp + fp + fn
        iou = tp / union if union > 0 else np.nan
        ious.append(iou)

    metrics = {
        "mean_iou": np.nanmean(ious),
        "mean_dice":  np.nanmean([2 * i/(1+i) if not np.isnan(i) else np.nan for i in ious])
    }
    for idx, i in enumerate(ious):
        metrics[f"iou_class_{idx}"] = i

    return metrics

def compute_metrics_wrapper(eval_pred):
    return compute_metrics(eval_pred,num_classes=5)


