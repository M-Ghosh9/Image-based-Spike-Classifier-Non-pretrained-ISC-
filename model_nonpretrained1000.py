# model_nonpretrained1000.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from torch import autocast, GradScaler
from torchvision import transforms
from torchvision.models import efficientnet_b3
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import csv

# Enable cuDNN autotuner for potential speedup
torch.backends.cudnn.benchmark = True

# Clear unnecessary GPU memory before starting
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class CustomDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        print(f"Initializing CustomDataset with file: {hdf5_file}")
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')  # Keep file open
        self.group_names = list(self.hf.keys())
        self.transform = transform

        # Filter data by classes 0-999
        self.filtered_indices = self._filter_data_by_classes()
        self.data_size = len(self.filtered_indices)
        print(f"Filtered data size: {self.data_size}")

    def _filter_data_by_classes(self):
        filtered_indices = []
        for group_name in self.group_names:
            group = self.hf[group_name]
            labels = group['labels'][:]
            for idx, label in enumerate(labels):
                if 0 <= label < 1000:  # Keep only labels 0-999
                    filtered_indices.append((group_name, idx))
        return filtered_indices

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        group_name, image_idx = self.filtered_indices[idx]
        group = self.hf[group_name]
        image = torch.tensor(group['images'][image_idx], dtype=torch.float32)
        label = torch.tensor(group['labels'][image_idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __del__(self):
        self.hf.close()

    def normalize_image(self, image):
        """Normalize image to the range [0, 1] for visualization."""
        return (image - image.min()) / (image.max() - image.min())

    def visualize_samples(self, num_samples=5, save_path=None):
        """Visualize and optionally save a few random samples from the dataset."""
        indices = random.sample(range(len(self)), num_samples)
        plt.figure(figsize=(12, 12))
        for i, idx in enumerate(indices):
            image, label = self[idx]
            image = image.numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
            image = self.normalize_image(image)  # Normalize the image
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Label: {label.item()}')
            plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_images_by_labels(self, labels):
        images = []
        for label in labels:
            indices = [i for i in range(len(self)) if self[i][1].item() == label]
            sampled_indices = random.sample(indices, min(len(indices), 5))
            for idx in sampled_indices:
                image, _ = self[idx]
                images.append((image, label))
        return images

def compute_class_weights(labels, num_classes=1000):
    print("Computing class weights...")
    # Count the frequency of each class in the labels
    class_sample_count = np.bincount(labels, minlength=num_classes)
    
    # Replace zeros with ones to avoid division by zero
    class_sample_count[class_sample_count == 0] = 1
    
    # Compute class weights
    class_weights = 1. / class_sample_count
    class_weights = class_weights / np.sum(class_weights)  # Normalize class weights
    
    return class_weights

def plot_class_weights(class_weights):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_weights)), class_weights)
    plt.xlabel('Class')
    plt.ylabel('Weight')
    plt.title('Class Weights Distribution')
    plt.tight_layout()
    plt.savefig('class_weights.png')
    plt.show()

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, debug=False):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.debug = debug
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        if self.debug:
            print(f"CBAM initialized with {channels} channels and reduction ratio of {reduction_ratio}.")

    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")

        # Channel attention
        avg_out = torch.mean(x, dim=[2, 3], keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # Max along height
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)  # Max along width
        avg_out = self.channel_attention(avg_out)
        max_out = self.channel_attention(max_out)
        x = x * (avg_out + max_out)
        return x

class EfficientNetCBAMPredictor(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetCBAMPredictor, self).__init__()
        # Initialize EfficientNet without pretrained weights
        self.efficientnet = efficientnet_b3(weights=None)
        in_features = self.efficientnet.classifier[1].in_features
        self.cbam = CBAM(channels=in_features)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        print(f"EfficientNetCBAMPredictor initialized for {num_classes} classes.")

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.cbam(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    print("Evaluating the model...")

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return loss, accuracy

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def save_checkpoint(state, filename='checkpoint_nonpretrained1000.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved as '{filename}'")

def load_checkpoint(filename='checkpoint_nonpretrained1000.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"No checkpoint found at {filename}")
        return None

def save_spike_images(dataset, labels, epoch):
    images = dataset.get_images_by_labels(labels)
    num_images = len(images)
    cols = 5
    rows = (num_images // cols) + (num_images % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    for img, lbl, ax in zip(images, labels, axes):
        image, label = img
        image = dataset.normalize_image(image.numpy().transpose(1, 2, 0))  # Normalize image
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f'spike_images_epoch_{epoch}.png')
    plt.close()

# Save metrics to a CSV file
def save_metrics_to_csv(train_losses, val_losses, train_accuracies, val_accuracies, filename='training_metricsNP.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch], train_accuracies[epoch], val_accuracies[epoch]])
    print(f"Training metrics saved to {filename}")

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data Augmentation
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        ])
        print("Data augmentation transforms applied.")

        # Dataset setup for classes 0-999
        dataset = CustomDataset('processed_images.h5', transform=transform)
        all_labels = [label for _, label in dataset.filtered_indices]  # Extract all labels
        class_weights = torch.tensor(compute_class_weights(all_labels, num_classes=1000), dtype=torch.float).to(device)
        num_classes = 1000  # We are focusing on classes 0-999
        print(f"Number of classes: {num_classes}")

        # Plot and save class weights
        plot_class_weights(class_weights.cpu().numpy())

        # Split dataset into training and validation sets
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size  # Ensure all data is used
        print(f"Train size: {train_size}, Validation size: {val_size}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print("Datasets split into training and validation.")

        # DataLoader with reduced batch size for memory management
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
        print("DataLoaders initialized.")

        # Visualize and save initial samples
        dataset.visualize_samples(num_samples=5, save_path='initial_samples.png')

        # Model setup without pretrained weights
        model = EfficientNetCBAMPredictor(num_classes=num_classes).to(device)
        print("Model setup complete.")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # More aggressive
        scaler = GradScaler()  # Corrected constructor
        print("Loss function, optimizer, and scheduler initialized.")

        # Load checkpoint if available
        checkpoint = load_checkpoint('checkpoint_nonpretrained1000.pth')
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_accuracy = checkpoint['best_val_accuracy']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_val_accuracy = -float('inf')

        # Training parameters
        epochs = 400  
        early_stop_patience = 5  # Early stopping patience
        early_stop_counter = 0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        checkpoint_accuracies = [20, 30, 40, 50, 60]  # Percentages at which to save model samples

        for epoch in range(start_epoch, epochs):
            print(f"\n{'-'*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            optimizer.zero_grad()
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Mixed precision training
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_dataloader)
            train_accuracy = 100. * correct / total

            val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Record metrics for learning curves
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Learning rate adjustment based on validation loss
            scheduler.step(val_loss)

            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
                print(f"New best accuracy: {best_val_accuracy:.2f}%. Saving checkpoint...")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies
                })
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

            torch.cuda.empty_cache()

        print("\nTraining complete. Saving final model checkpoint...")
        save_checkpoint({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        })

        # Save metrics to a CSV file
        save_metrics_to_csv(train_losses, val_losses, train_accuracies, val_accuracies)

        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        print("Learning curves plotted.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        torch.cuda.empty_cache()

