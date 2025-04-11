import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# ========== STEP 1: Load and Preprocess Data ==========

def preprocess_csi_for_1dcnn(data):
    X = []
    for i in range(data.shape[2]):
        sample = np.abs(data[:, :, i])
        norm = sample / (sample[0:1, :] + 1e-8)
        flat = norm.T.flatten()
        X.append(flat)
    return np.array(X)

presence = np.load('dataset_SDR/dataPresence.npy')
no_presence = np.load('dataset_SDR/NoPresence.npy')
small_presence = np.load('dataset_SDR/dataSmallPresence.npy')

X_presence = preprocess_csi_for_1dcnn(presence)
X_no_presence = preprocess_csi_for_1dcnn(no_presence)
X_small_presence = preprocess_csi_for_1dcnn(small_presence)

# Labels: 0 - presence, 1 - no presence, 2 - small presence
y_presence = np.full((X_presence.shape[0],), 0)
y_no_presence = np.full((X_no_presence.shape[0],), 1)
y_small_presence = np.full((X_small_presence.shape[0],), 2)

X = np.vstack([X_presence, X_no_presence, X_small_presence])
y = np.concatenate([y_presence, y_no_presence, y_small_presence])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split 70% train, 15% val, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ========== STEP 2: Dataset & DataLoader ==========

class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CSIDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=32)

# ========== STEP 3: 1D CNN Model ==========

class CSI1DCNN(nn.Module):
    def __init__(self):
        super(CSI1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        dummy_input = torch.zeros(1, 1, 416)
        out = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        self.flattened_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== STEP 4: Train Loop ==========

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
model = CSI1DCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

print("\nTraining model...\n")
for epoch in range(50):
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
    train_loss_history.append(train_loss)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {train_loss:.4f} | Time: {time.time()-start:.2f}s")

# ========== STEP 5: Final Evaluation ==========

model.eval()
correct, total = 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\n Final Test Accuracy: {accuracy:.4f}")

# ========== STEP 6: Classification Report ==========

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Presence', 'No Presence', 'Small Presence']))

# ========== STEP 7: Confusion Matrix in Percentage ==========

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percent = np.round(cm_normalized * 100, 2)

labels = ['Presence', 'No Presence', 'Small Presence']
df_cm = pd.DataFrame(cm_percent, index=labels, columns=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.title('Confusion Matrix (%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ========== STEP 8: Plot Accuracy Trends ==========

plt.figure()
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training & Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
