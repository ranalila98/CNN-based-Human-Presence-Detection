import torch
import torch.nn as nn
import torch.nn.functional as F

class CSI1DCNN(nn.Module):
    def __init__(self, input_len):
        super(CSI1DCNN, self).__init__()

        # Convolutional layers with different kernel size and padding
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2) 
        # self.conv1 = nn.Conv1d(1, 16, kernel_size=5) 
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(0.3)  

        # Dummy forward pass to calculate flattened size
        dummy_input = torch.zeros(1, 1, input_len)
        with torch.no_grad():
            out = self.pool1(F.relu(self.conv1(dummy_input)))
            out = self.pool2(F.relu(self.conv2(out)))
            out = self.dropout1(out)
            self.flattened_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.dropout2 = nn.Dropout(0.3)  # after first FC layer
        self.fc2 = nn.Linear(64, 3)  # 3 output classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)
