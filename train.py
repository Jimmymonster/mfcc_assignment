import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mfcc import mfcc
import os
import json
from model import MFCCDataset, CNNLSTMEmotionModel

# Training and Validation Function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Training Epoch [{epoch+1}/{num_epochs}]")
        model.train()  # Set the model to training mode
        total_loss = 0
        correct = 0
        total = 0

        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

        # Validation after each epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_correct/val_total:.4f}")

# Load Dataset and Train
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Extract required information
    emo_dict = {name: emotion_mapping[details[0]["assigned_emo"]] for name, details in data.items()}

    # Load file paths
    datasets = os.listdir("dataset")
    for dataset in datasets:
        if os.path.isfile(f"dataset/{dataset}"):
            continue
        studios = os.listdir(f"dataset/{dataset}")
        for studio in studios:
            sub_files = ["con"]
            for sub_file in sub_files:
                file_paths = os.listdir(f"dataset/{dataset}/{studio}/{sub_file}")
                for file_path in file_paths:
                    file_name = f"dataset/{dataset}/{studio}/{sub_file}/{file_path}"
                    if file_path in emo_dict:  # Ensure matching label exists
                        file_paths_input.append(file_name)
                        labels_input.append(emo_dict[file_path])

    # Create the full dataset
    dataset = MFCCDataset(file_paths_input, labels_input)

    # Split dataset into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Choose model
    model = CNNLSTMEmotionModel(num_classes=5).to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10)
