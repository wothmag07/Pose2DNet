import torch
import numpy as np
import matplotlib.pyplot as plt

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l

def calculate_precision_recall(pred_keypoints, true_keypoints, threshold=10):
    """
    Calculate precision and recall based on predicted keypoints and true keypoints.
    
    Args:
        pred_keypoints (torch.Tensor): Predicted keypoints of shape (batch_size, num_keypoints, 2).
        true_keypoints (torch.Tensor): True keypoints of shape (batch_size, num_keypoints, 2).
        threshold (float): Threshold distance to consider a prediction as correct.

    Returns:
        precision (float): Precision value.
        recall (float): Recall value.
    """
    batch_size, num_keypoints, _ = pred_keypoints.size()
    
    # Calculate distances between predicted and true keypoints
    distances = torch.norm(pred_keypoints - true_keypoints, dim=2)
    
    # Count true positives, false positives, and false negatives
    true_positives = (distances < threshold).sum().item()
    false_positives = ((distances < threshold).sum(dim=1) - 1).clamp(min=0).sum().item()
    false_negatives = ((distances.min(dim=1).values >= threshold).sum()).item()

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def calculate_mAP(precision, recall):
    # Calculate Mean Average Precision (mAP)
    # Precision and recall should be tensors of shape (batch_size,)
    ap = (precision * recall).sum() / recall.sum()
    return ap

def calculate_oks(pred_keypoints, true_keypoints, threshold=0.5):
    # Calculate Object Keypoint Similarity (OKS)
    # pred_keypoints and true_keypoints should be of shape (batch_size, num_keypoints, 2)
    diff = pred_keypoints - true_keypoints
    squared_error = torch.sum(diff ** 2, dim=2)
    distance = torch.sqrt(squared_error)
    oks = torch.exp(-distance / (2 * threshold**2))
    return oks.mean()

def calculate_pck(pred_keypoints, true_keypoints, threshold=10):
    # Calculate Percentage of Correct Keypoints (PCK)
    # pred_keypoints and true_keypoints should be of shape (batch_size, num_keypoints, 2)
    diff = torch.norm(pred_keypoints - true_keypoints, dim=2)
    correct_keypoints = (diff < threshold).sum(dim=1)
    pck = correct_keypoints.float().mean() / true_keypoints.size(1)
    return pck

def plot_metrics(train_losses, val_losses, pck_values, oks_values, precision_values, recall_values, mAP_values):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Losses')
    plt.legend()
    plt.savefig('train_loss_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.legend()
    plt.savefig('val_loss_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(pck_values, label='PCK')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('PCK')
    plt.legend()
    plt.savefig('pck_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(oks_values, label='OKS')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('OKS')
    plt.legend()
    plt.savefig('oks_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(precision_values, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Precision')
    plt.legend()
    plt.savefig('precision_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(recall_values, label='Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Recall')
    plt.legend()
    plt.savefig('recall_plot.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.plot(mAP_values, label='mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()
    plt.savefig('mAP_plot.png')
    plt.close()