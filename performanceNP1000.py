# performanceNP1000.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from model_nonpretrained1000 import EfficientNetCBAMPredictor
from matplotlib.backends.backend_pdf import PdfPages
import time
import h5py
import random
import pandas as pd

# Add timing logs
def time_log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Custom Dataset class for filtered data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, transform=None):
        time_log(f"Initializing CustomDataset with file: {hdf5_file}")
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')  # Keep file open
        self.group_names = list(self.hf.keys())
        self.transform = transform

        # Filter data by classes 0-999
        self.filtered_indices = self._filter_data_by_classes()
        self.data_size = len(self.filtered_indices)
        time_log(f"Filtered data size: {self.data_size}")

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

# Load learning curves from CSV file
def load_learning_curves_from_csv(filename='training_metricsNP.csv'):
    time_log(f"Loading learning curves from {filename}...")
    df = pd.read_csv(filename)
    return {
        'train_losses': df['Train Loss'].tolist(),
        'val_losses': df['Validation Loss'].tolist(),
        'train_accuracies': df['Train Accuracy'].tolist(),
        'val_accuracies': df['Validation Accuracy'].tolist()
    }

# Plot learning curves from training history
def plot_learning_curves(history, pdf=None):
    time_log("Plotting learning curves...")
    epochs = np.arange(1, len(history['train_losses']) + 1)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accuracies = history['train_accuracies']
    val_accuracies = history['val_accuracies']

    plt.figure(figsize=(12, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve for Non-Pretrained Model')
    plt.grid(True)

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve for Non-Pretrained Model')
    plt.grid(True)

    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.close()

# Plot Top-K Accuracy
def plot_top_k_accuracy(accuracies, k_values=[1, 5, 10], pdf=None):
    time_log("Plotting top-k accuracy...")
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o', color='purple')
    plt.title("Top-K Accuracy for Non-Pretrained Model")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    if pdf:
        pdf.savefig()
    plt.close()

# Plot Random Sample Images with Predictions
def plot_random_samples(images, true_labels, predicted_labels, label_encoder, pdf=None, num_samples=10):
    time_log(f"Plotting {num_samples} random samples...")
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    sample_indices = random.sample(range(len(true_labels)), num_samples)
    
    for i, ax in enumerate(axes.flatten()):
        idx = sample_indices[i]
        image = images[idx].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        true_class = label_encoder.inverse_transform([true_labels[idx]])[0]
        pred_class = label_encoder.inverse_transform([predicted_labels[idx]])[0]
        title = f"True: {true_class}\nPred: {pred_class}"
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.close()

if __name__ == "__main__":
    time_log("Starting script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_log(f"Using device: {device}")

    # Create dataset and dataloader for classes 0-999
    dataset = CustomDataset('processed_images.h5')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    time_log("Dataloader created.")

    # Load the non-pretrained model with the correct number of classes
    num_classes = 1000
    model = EfficientNetCBAMPredictor(num_classes).to(device)
    time_log(f"EfficientNetCBAMPredictor initialized for {num_classes} classes.")

    # Load the checkpoint
    checkpoint = torch.load('checkpoint_nonpretrained1000.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    time_log("Model weights loaded and evaluation mode set.")

    # Load training history from CSV
    history = load_learning_curves_from_csv('training_metricsNP.csv')

    # Initialize accumulators
    top_k_accumulator = {k: 0 for k in [1, 5, 10]}
    all_labels = []
    all_predictions = []

    # For random sample visualization
    random_images, random_labels, random_predictions = [], [], []

    time_log("Starting predictions...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()

            # Flatten labels to 1D
            labels_np = labels.cpu().numpy().flatten()
            probs_np = probs

            # Accumulate top-k accuracy
            for k in top_k_accumulator.keys():
                top_k_accumulator[k] += top_k_accuracy_score(
                    labels_np, 
                    probs_np, 
                    k=k,
                    labels=range(num_classes)  # Ensure all classes are included
                ) * len(labels_np)

            # Accumulate true labels and predictions for final report
            all_labels.extend(labels_np)
            all_predictions.extend(predictions)

            # Collect random samples for visualization
            if batch_idx == 0:  # Collect from the first batch for simplicity
                random_images = inputs.cpu()
                random_labels = labels.cpu().numpy()
                random_predictions = predictions

            if batch_idx % 500 == 0:
                time_log(f"Processed batch {batch_idx} / {len(dataloader)}")

        # Finalize top-k accuracy calculation
        for k in top_k_accumulator.keys():
            top_k_accumulator[k] /= len(dataset)

        time_log(f"Final top-k accuracies: {top_k_accumulator}")

        # Classification Report
        report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(num_classes)], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv('classification_report_summary_NP1000.csv', index=True)
        time_log("Classification report saved to 'classification_report_summary_NP1000.csv'.")

        # Plot and save performance results in a PDF
        with PdfPages('NP1000.pdf') as pdf:
            plot_learning_curves(history, pdf=pdf)
            plot_top_k_accuracy([top_k_accumulator[k] for k in [1, 5, 10]], pdf=pdf)
            plot_random_samples(random_images, random_labels, random_predictions, LabelEncoder().fit([str(i) for i in range(num_classes)]), pdf=pdf)

    time_log("Script finished.")

