import torch
from torch.utils.data import Dataset
from models.cnn_model import CSI1DCNN

def load_trained_model(path, input_len, device=None):
    model = CSI1DCNN(input_len=input_len)
    if device:
        model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]