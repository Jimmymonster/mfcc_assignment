import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import MFCCDataset, CNNLSTMEmotionModel

# Evaluation Function with Confusion Matrix
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

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Load Dataset and Evaluate
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset paths and labels (same as the original training code)
    file_paths_input = []
    labels_input = []
    with open("dataset/emotion_label.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    emotion_mapping = {
        "Neutral": 0,
        "Angry": 1,
        "Happy": 2,
        "Sad": 3,
        "Frustrated": 4,
    }

    emo_dict = {name: emotion_mapping[details[0]["assigned_emo"]] for name, details in data.items()}

    # Load file paths for evaluation set
    val_file_paths_input = []  # Add a separate file paths for validation data
    val_labels_input = []      # Add corresponding labels for validation data

    # Assuming `validation` folder exists in the dataset structure
    datasets = os.listdir("dataset")
    for dataset in datasets:
        if os.path.isfile(f"dataset/{dataset}"):
            continue
        studios = os.listdir(f"dataset/{dataset}")
        for studio in studios:
            sub_files = ["con"]  # Change to validation
            for sub_file in sub_files:
                file_paths = os.listdir(f"dataset/{dataset}/{studio}/{sub_file}")
                for file_path in file_paths:
                    file_name = f"dataset/{dataset}/{studio}/{sub_file}/{file_path}"
                    if file_path in emo_dict:  # Ensure matching label exists
                        val_file_paths_input.append(file_name)
                        val_labels_input.append(emo_dict[file_path])

    val_dataset = MFCCDataset(val_file_paths_input, val_labels_input)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load trained model (assuming it's saved)
    model = CNNLSTMEmotionModel(num_classes=5).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))
    
    # Evaluate model
    confusion_mat = evaluate_model(model, val_loader, device)

    # Class names
    class_names = ["Neutral", "Angry", "Happy", "Sad", "Frustrated"]

    # Plot confusion matrix
    plot_confusion_matrix(confusion_mat, class_names)
