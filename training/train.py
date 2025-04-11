import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from models.cnn_model import CSI1DCNN

def train_model(model, train_loader, val_loader, device=None, epochs=30, lr=0.01):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        print(" Using CPU")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_acc_history = []
    val_acc_history = []

    print(" Training model...")
    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {train_loss:.4f} | Time: {time.time()-start:.2f}s")

    # Plot and save Accuracy graph
    plt.figure()
    plt.plot(train_acc_history, 'g--', label='Train Accuracy')
    plt.plot(val_acc_history,'b--', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = 'plots/accuracy_plot.png'
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to {plot_path}")
    plt.close()

    # Save trained model
    save_path = 'checkpoints/best_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

from sklearn.metrics import confusion_matrix, accuracy_score
from testing.kfold_confusion import plot_avg_confusion_matrix_kfold  # adjust import path

def k_fold_cross_validation(X, y, folds=5, epochs=20, lr=0.01, batch_size=32):
    import random
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_accuracies = []

    all_y_true_folds = []
    all_y_pred_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n Fold {fold + 1}/{folds}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.long))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CSI1DCNN(input_len=X.shape[1]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(labels.numpy())

        acc = accuracy_score(all_true, all_preds)
        print(f" Fold {fold + 1} Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)
        all_y_true_folds.append(np.array(all_true))
        all_y_pred_folds.append(np.array(all_preds))

    # Plot averaged K-Fold confusion matrix
    class_names = ['Presence', 'No Presence', 'Small Presence']
    plot_avg_confusion_matrix_kfold(
        all_y_true_folds,
        all_y_pred_folds,
        class_names,
        save_path='plots/kfold_avg_confusion.png'
    )

    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n Average Accuracy over {folds} folds: {avg_acc:.4f} ± {std_acc:.4f}")

    return fold_accuracies
