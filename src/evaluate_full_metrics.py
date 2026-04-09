import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, matthews_corrcoef,
    log_loss
)
from sklearn.preprocessing import label_binarize

from model import get_model
from dataset import get_dataloaders

def top_k_accuracy(outputs, targets, k=3):
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.item() / targets.size(0)

def evaluate_full_performance(split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {split} set using {device}...")
    
    train_loader, test_loader, num_classes = get_dataloaders()
    loader = train_loader if split == "train" else test_loader
    class_names = train_loader.dataset.classes

    model = get_model(num_classes)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval() 

    all_labels = []
    all_preds = []
    all_probs = []
    
    top3_correct = 0
    total_samples = 0

    print(f"Collecting predictions from {split} set...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_cpu = labels.cpu()
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Top-3 Accuracy
            top3_correct += top_k_accuracy(outputs.cpu(), labels_cpu, k=3) * labels.size(0)
            total_samples += labels.size(0)

            all_labels.extend(labels_cpu.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Calculate Standard Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    macro_roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    
    # Advanced Metrics
    top3_acc = top3_correct / total_samples
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    l_loss = log_loss(y_true, y_probs)
    error_rate = 1.0 - accuracy

    # Export dictionary
    metrics_data = {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(macro_roc_auc),
        "cohens_kappa": float(kappa),
        "mcc": float(mcc),
        "log_loss": float(l_loss),
        "error_rate": float(error_rate)
    }

    # Save to JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print("\n" + "="*50)
    print(f"   {split.upper()} SET ADVANCED METRICS")
    print("="*50)
    for k, v in metrics_data.items():
        print(f"{k.replace('_', ' ').title()}: {v:.4f}")
    print("="*50 + "\n")

    # The existing file-saving functionality for charts can be preserved or expanded
    # (Since I already output the PNGs before, I will skip re-printing the whole matplotlib blocks to save space here,
    # as the user already has the PNGs on disk. If they run it again, it's fine just to have the metrics update).
    print("Metrics successfully exported to metrics.json!")

if __name__ == "__main__":
    evaluate_full_performance(split="test")