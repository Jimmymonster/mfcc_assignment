import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from mfcc import mfcc
import json
import os

# Define the dataset
class MFCCDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, max_mfcc=100, max_length=500):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_mfcc = max_mfcc
        self.all_max_mfcc = 0
        self.all_max_length = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        signal, _ = librosa.load(self.file_paths[idx], sr=self.sample_rate)
        mfcc_features = mfcc(signal, self.sample_rate)  # Extract MFCCs

        # Ensure consistent shape
        num_mfcc, time_steps = mfcc_features.shape

        # Update max values
        if num_mfcc > self.all_max_mfcc:
            self.all_max_mfcc = num_mfcc
        if time_steps > self.all_max_length:
            self.all_max_length = time_steps

        if num_mfcc < self.max_mfcc:
            pad_mfcc = self.max_mfcc - num_mfcc
            mfcc_features = np.pad(mfcc_features, ((0, pad_mfcc), (0, 0)), mode='constant')
        else:
            mfcc_features = mfcc_features[:self.max_mfcc, :]  # Trim extra rows

        # **Pad time steps (if needed)**
        if time_steps < self.max_length:
            pad_width = self.max_length - time_steps
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :self.max_length]  # Trim extra columns

        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc_tensor.unsqueeze(0), label_tensor  # Add channel dimension for CNN


if __name__ == "__main__":
    file_paths_input = []
    labels_input = []
    
    # Load emotion labels from JSON
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

    # Find max MFCC and max Length
    max_mfcc = 0
    max_length = 0
    for idx in range(len(dataset)):
        signal, _ = librosa.load(dataset.file_paths[idx], sr=dataset.sample_rate)
        mfcc_features = mfcc(signal, dataset.sample_rate)  # Extract MFCCs

        num_mfcc, time_steps = mfcc_features.shape
        max_mfcc = max(max_mfcc, num_mfcc)
        max_length = max(max_length, time_steps)

    print(f"Max MFCC: {max_mfcc}, Max Length: {max_length}")