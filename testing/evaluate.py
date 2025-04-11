import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_model(model, test_loader, device=None):
    os.makedirs('plots', exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device is: {device}")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    print(f"\n Test Accuracy: {accuracy:.4f}")

    print("\n Classification Report:")
    target_names = ['Presence', 'No Presence', 'Small Presence']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.round(cm_normalized * 100, 2)

    df_cm = pd.DataFrame(cm_percent, index=target_names, columns=target_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix ')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = 'plots/confusion_matrix.png'
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")
    plt.close()



