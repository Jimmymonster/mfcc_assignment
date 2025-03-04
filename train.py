import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import json
from model import MFCCDataset, CNNLSTMEmotionModel
from torch.utils.tensorboard import SummaryWriter

# Training and Validation Function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, batch_size=16, writer=None, run_path="runs"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create run_path and subdirectories for checkpoints and logs if they don't exist
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    log_dir = os.path.join(run_path, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load checkpoint if available
    checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
    if os.path.exists(checkpoint_path):
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

        for mfccs, labels in train_loader:
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
        with torch.no_grad():
            for mfccs, labels in val_loader:
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

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Save the best models based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"Best model saved with val loss {best_val_loss:.4f}")

        # Track the latest models
        latest_model_path = os.path.join(checkpoint_dir, f"latest_model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_model_path)
        print(f"Latest model saved for epoch {epoch+1}")

        latest_models.append(latest_model_path)
        if len(latest_models) > 3:
            oldest_model = latest_models.pop(0)
            os.remove(oldest_model)  # Remove the oldest model file

        # Save the last 3 best models based on validation loss
        if len(best_models) < 3:
            best_models.append((avg_val_loss, epoch + 1))
            # Sort models based on validation loss (ascending)
            best_models = sorted(best_models, key=lambda x: x[0])
        else:
            # If we have 3 models, check if we need to add the current model
            if avg_val_loss < best_models[-1][0]:
                best_models[-1] = (avg_val_loss, epoch + 1)
                best_models = sorted(best_models, key=lambda x: x[0])

        # Save the top 3 best models
        for val_loss, epoch_num in best_models:
            model_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch_num}.pth")
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss,
            }, model_checkpoint_path)
            print(f"Best model from epoch {epoch_num} with val loss {val_loss:.4f} saved")

    # Close the writer when done
    writer.close()

# Load Dataset and Train
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set run_path dynamically (e.g., current timestamp or any other identifier)
    run_path = os.path.join("runs", "run_1")  # Modify "run_1" based on your system's timestamp or identifier
    dataset_path = "dataset_tmp"  # Modify this path based on your dataset location
    batch_size = 32
    lr=0.0001

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

    # Load file paths
    datasets = os.listdir(dataset_path)
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
      # You can modify this batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check for existing checkpoint to resume training
    model = CNNLSTMEmotionModel(num_classes=5).to(device)
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
    train_model(model, train_loader, val_loader, num_epochs=50, lr=lr , batch_size=batch_size, writer=writer, run_path=run_path)

    # Close the writer when done
    writer.close()
