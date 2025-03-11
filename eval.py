import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import MFCCDataset, LSTMEmotionModel
from sklearn.metrics import classification_report

# Create output directory
output_dir = "validation_result"
os.makedirs(output_dir, exist_ok=True)

def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for mfccs, labels in val_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_labels, all_preds

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_paths_input = []
    labels_input = []
    with open("dataset/emotion_label.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    emotion_mapping = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Frustrated": 4}
    emo_dict = {name: emotion_mapping[details[0]["assigned_emo"]] for name, details in data.items()}

    val_file_paths_input = []
    val_labels_input = []

    datasets = os.listdir("dataset")
    for dataset in datasets:
        if os.path.isfile(f"dataset/{dataset}"):
            continue
        studios = os.listdir(f"dataset/{dataset}")
        if dataset not in ["studio51-60"]:
            continue
        for studio in studios:
            sub_files = ["clip", "con", "middle"]
            for sub_file in sub_files:
                file_paths = os.listdir(f"dataset/{dataset}/{studio}/{sub_file}")
                for file_path in file_paths:
                    file_name = f"dataset/{dataset}/{studio}/{sub_file}/{file_path}"
                    if file_path in emo_dict:
                        val_file_paths_input.append(file_name)
                        val_labels_input.append(emo_dict[file_path])

    val_dataset = MFCCDataset(val_file_paths_input, val_labels_input, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = LSTMEmotionModel(num_classes=5).to(device)
    model.load_state_dict(torch.load("/home/jim/emotion_detection/runs/run_12/checkpoints/best_model.pth"))
    
    confusion_mat, all_labels, all_preds = evaluate_model(model, val_loader, device)
    class_names = ["Neutral", "Angry", "Happy", "Sad", "Frustrated"]

    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(confusion_mat, class_names, cm_path)

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Save report to a text file
    report_path = "validation_result/classification_report.txt"
    os.makedirs("validation_result", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report saved to {report_path}")
    
    # Save confusion matrix as JSON
    cm_json_path = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_json_path, "w") as f:
        json.dump(confusion_mat.tolist(), f)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    results = {
        "true_labels": [int(label) for label in all_labels], 
        "predicted_labels": [int(pred) for pred in all_preds]
    }
    with open(results_path, "w") as f:
        json.dump(results, f)
    
    print(f"Results saved in {output_dir}")
