import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_avg_confusion_matrix_kfold(all_y_true_folds, all_y_pred_folds, class_names, save_path=None):
    assert len(all_y_true_folds) == len(all_y_pred_folds), "Mismatch in fold count"

    n_classes = len(class_names)
    total_cm = np.zeros((n_classes, n_classes), dtype=np.float32)

    for y_true, y_pred in zip(all_y_true_folds, all_y_pred_folds):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        total_cm += cm

    avg_cm = total_cm / len(all_y_true_folds)
    avg_cm_percent = avg_cm / avg_cm.sum(axis=1, keepdims=True) * 100
    df_cm = pd.DataFrame(np.round(avg_cm_percent, 2), index=class_names, columns=class_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix over K-Folds', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved average K-Fold confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()
