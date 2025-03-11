import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from mfcc import mfcc

# Define the dataset
class MFCCDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, n_mfcc=4096, max_length=100, 
                 augment=False, add_noise=True, mask=True):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc  # Reduce MFCC features to 40 (not 8192)
        self.max_length = max_length  # Time steps
        self.augment = augment
        self.add_noise = add_noise
        self.mask = mask

    def __len__(self):
        return len(self.file_paths)

    def augment_mfcc(self, mfcc):
        """Apply augmentation techniques on the MFCC features"""
        if self.add_noise and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, mfcc.shape)  # Gaussian noise
            mfcc += noise

        if self.mask and np.random.rand() < 0.5:
            # Time Masking (Randomly mask out time steps)
            time_mask = np.random.randint(1, self.max_length // 4)
            t_start = np.random.randint(0, self.max_length - time_mask)
            mfcc[:, t_start:t_start + time_mask] = 0
            
            # Frequency Masking (Randomly mask out frequency bins)
            freq_mask = np.random.randint(1, self.n_mfcc // 4)
            f_start = np.random.randint(0, self.n_mfcc - freq_mask)
            mfcc[f_start:f_start + freq_mask, :] = 0
        
        return mfcc

    def __getitem__(self, idx):
        # Load audio
        signal, _ = librosa.load(self.file_paths[idx], sr=self.sample_rate)

        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc)

        # Normalize MFCC features
        mfcc_features = (mfcc_features - np.mean(mfcc_features)) / (np.std(mfcc_features) + 1e-6)

        # Apply augmentations to MFCC
        if self.augment:
            mfcc_features = self.augment_mfcc(mfcc_features)

        # Get time steps
        _, time_steps = mfcc_features.shape

        # **Fix: Ensure fixed shape (n_mfcc, max_length)**
        if time_steps < self.max_length:
            pad_width = self.max_length - time_steps
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :self.max_length]

        # Convert to tensor and reshape for LSTM input
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).transpose(0, 1)  # Shape: (time_steps, n_mfcc)

        # Convert label to tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mfcc_tensor, label_tensor
# LSTM Emotion Model
# class LSTMEmotionModel(nn.Module):
#     def __init__(self, num_classes):
#         super(LSTMEmotionModel, self).__init__()

#         # LSTM Layers following the decreasing structure
#         self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
#         self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
#         self.lstm4 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
#         self.lstm5 = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)

#         # Fully connected output layer
#         self.fc = nn.Linear(8, num_classes)

#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x, _ = self.lstm3(x)
#         x, _ = self.lstm4(x)
#         x, _ = self.lstm5(x)

#         # Take the last time step's output
#         x = x[:, -1, :]
#         x = self.fc(x)
#         return x

class LSTMEmotionModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(LSTMEmotionModel, self).__init__()

        # LSTM Layers with bidirectional=True
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=16, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(input_size=32, hidden_size=8, batch_first=True, bidirectional=True)

        # Layer normalization
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(64)
        self.norm4 = nn.LayerNorm(32)
        self.norm5 = nn.LayerNorm(16)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(16, num_classes)  # Adjusted for bidirectional LSTM output

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x, _ = self.lstm3(x)
        x = self.norm3(x)
        x, _ = self.lstm4(x)
        x = self.norm4(x)
        x, _ = self.lstm5(x)
        x = self.norm5(x)

        # Take the last time step's output
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)

        return x