import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from mfcc import mfcc

# Define the dataset
class MFCCDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, max_mfcc=7500, max_length=12):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_mfcc = max_mfcc

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        signal, _ = librosa.load(self.file_paths[idx], sr=self.sample_rate)
        mfcc_features = mfcc(signal, self.sample_rate)  # Extract MFCCs

        # Ensure consistent shape
        num_mfcc, time_steps = mfcc_features.shape
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

# CNN-LSTM Hybrid Model
class CNNLSTMEmotionModel(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2):
        super(CNNLSTMEmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, mfcc, time)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, reduced_mfcc, reduced_time)

        # Flatten the spatial dimensions into a sequence dimension
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height * width, channels)  # (batch, seq_len, feature_dim)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last hidden state
        x = self.fc(x)
        return x