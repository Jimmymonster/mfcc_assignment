import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
from model import MFCCDataset, LSTMEmotionModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # Import tqdm for progress bars

# Training and Validation Function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, batch_size=16, writer=None, run_path="runs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=50) # not using it anymore

    # Create run_path and subdirectories for checkpoints and logs if they don't exist
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    log_dir = os.path.join(run_path, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load checkpoint if available
    checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
    if os.path.exists(checkpoint_path):
        print("found checkpoint, loading model from checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    # Create SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Lists to track best and latest models
    best_models = []
    latest_models = []

    for epoch in range(start_epoch, num_epochs):
        print(f"Training Epoch [{epoch+1}/{num_epochs}]")
        model.train()  # Set the model to training mode
        total_loss = 0
        correct = 0
        total = 0

        # Wrap train_loader with tqdm to show progress
        for mfccs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch", mininterval=30):
            mfccs, labels = mfccs.to(device), labels.to(device)
            
            outputs = model(mfccs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Log training loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Validation after each epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0

        # Wrap val_loader with tqdm for progress bar
        with torch.no_grad():
            for mfccs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", unit="batch", mininterval=30):
                mfccs, labels = mfccs.to(device), labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"lr = {scheduler.get_last_lr()[0]}")
        
        scheduler.step(avg_val_loss)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        # Save the best models based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save only model's state_dict for the best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val loss {best_val_loss:.4f}")
            # Save the last 5 best models based on validation loss
            if len(best_models) < 5:
                best_models.append((avg_val_loss, epoch + 1, best_model_path))
                # Sort models based on validation loss (ascending)
                best_models = sorted(best_models, key=lambda x: x[0])
            else:
                # Replace the worst model with the new one
                worst_model = best_models.pop(-1)
                worst_model_path = os.path.join(checkpoint_dir,f"best_model_epoch_{worst_model[1]}.pth")
                if os.path.exists(worst_model_path):
                    os.remove(worst_model_path)
                best_models.append((avg_val_loss, epoch + 1, best_model_path))
                
                # Sort models based on validation loss (ascending)
                best_models = sorted(best_models, key=lambda x: x[0])

            # Save the top 3 best models (only model state_dict)
            for val_loss, epoch_num, model_path in best_models:
                model_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch_num}.pth")
                torch.save(model.state_dict(), model_checkpoint_path)
                print(f"Best model from epoch {epoch_num} with val loss {val_loss:.4f} saved")
            print("")

        # Track the latest models
        latest_model_path = os.path.join(checkpoint_dir, f"latest_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), latest_model_path)  # Save only model's state_dict for latest model
        print(f"Latest model saved for epoch {epoch+1}")

        latest_models.append(latest_model_path)
        if len(latest_models) > 3:
            oldest_model = latest_models.pop(0)
            os.remove(oldest_model)  # Remove the oldest model file


        # Save the checkpoint with all information
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1}")

    # Close the writer when done
    writer.close()

# Load Dataset and Train
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set run_path dynamically (e.g., current timestamp or any other identifier)
    run_path = os.path.join("runs", "run_12")  # Modify "run_1" based on your system's timestamp or identifier
    dataset_path = "dataset"  # Modify this path based on your dataset location
    batch_size = 32
    lr=0.000005
    num_epochs=500

    # Create the run_path directory if it doesn't exist
    os.makedirs(run_path, exist_ok=True)

    # Create SummaryWriter instance for TensorBoard
    log_dir = os.path.join(run_path, "logs")
    writer = SummaryWriter(log_dir=log_dir)

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
    agree_dict = {name: details[0]["agreement"] for name, details in data.items()}

    # Load file paths
    datasets = os.listdir(dataset_path)
    for dataset in datasets:
        if os.path.isfile(f"dataset/{dataset}"):
            continue
        studios = os.listdir(f"dataset/{dataset}")

        for studio in studios:
            sub_files = ["con", "clip", "middle"]
            for sub_file in sub_files:
                sub_file_path = f"dataset/{dataset}/{studio}/{sub_file}"
                if not os.path.isdir(sub_file_path):
                    continue

                file_paths = os.listdir(sub_file_path)
                for file_path in file_paths:
                    file_name = os.path.join(sub_file_path, file_path)
                    if file_path in emo_dict and agree_dict[file_path] > 0.75:
                        file_paths_input.append(file_name)
                        labels_input.append(emo_dict[file_path])  # Collect labels for stratified splitting

    # Convert lists to numpy arrays for stratified splitting
    file_paths_input = np.array(file_paths_input)
    labels_input = np.array(labels_input)

    # Perform a stratified split (80% training, 20% validation)
    train_indices, val_indices = train_test_split(
        np.arange(len(labels_input)),  # Indices of dataset
        test_size=0.2,
        stratify=labels_input,  # Preserve class distribution
        random_state=42  # Reproducibility
    )

    # Create datasets with proper augmentation
    full_dataset = MFCCDataset(file_paths_input, labels_input, augment=True)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(MFCCDataset(file_paths_input, labels_input, augment=False), val_indices)

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check for existing checkpoint to resume training
    model = LSTMEmotionModel(num_classes=5).to(device)
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Ensure requires_grad=True for all parameters
        for param in m.parameters():
            param.requires_grad = True

    model.apply(init_weights)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr , batch_size=batch_size, writer=writer, run_path=run_path)

    # Close the writer when done
    writer.close()
