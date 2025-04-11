from preprocessing.preprocess import load_and_preprocess_data
from utils.loader import CSIDataset
from models.cnn_model import CSI1DCNN
from training.train import train_model, k_fold_cross_validation
from testing.evaluate import evaluate_model

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if hasattr(torch, "use_deterministic_algorithms"):
    #     torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    set_seed(42)

    print("\n Loading and preprocessing CSI data...")
    X, y = load_and_preprocess_data()

    print("\n Splitting dataset...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )
    print("Train Data Size:", X_train.shape)
    print("Test Data Size:", X_test.shape)
    print("Validation Data Size:", X_val.shape)

    # Dataloaders with fixed worker seed
    def seed_worker(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(
        CSIDataset(X_train, y_train), batch_size=32, shuffle=True, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        CSIDataset(X_val, y_val), batch_size=32, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        CSIDataset(X_test, y_test), batch_size=32, worker_init_fn=seed_worker
    )

    print("\n Training model (Train/Val split)...")
    model = CSI1DCNN(input_len=X.shape[1])
    model = train_model(model, train_loader, val_loader)

    print("\n Evaluating on test set...")
    evaluate_model(model, test_loader)

    print("\n Running K-Fold Cross-Validation...")
    fold_accuracies = k_fold_cross_validation(X, y, folds=5, epochs=10)

    print("\n K-Fold Accuracies:")
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1}: {acc:.4f}")

    print(f"\n Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    print("\n Done.")
