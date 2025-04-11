
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter

def preprocess_csi_for_1dcnn(data):
    X = []
    for i in range(data.shape[2]):
        sample = np.abs(data[:, :, i])
        norm = sample / (sample[0:1, :] + 1e-8)
        flat = norm.T.flatten()
        X.append(flat)
    return np.array(X)

def load_and_preprocess_data():
    presence = np.load('dataset_SDR/dataPresence.npy')
    no_presence = np.load('dataset_SDR/NoPresence.npy')
    small_presence = np.load('dataset_SDR/dataSmallPresence.npy')

    X1 = preprocess_csi_for_1dcnn(presence)
    X2 = preprocess_csi_for_1dcnn(no_presence)
    X3 = preprocess_csi_for_1dcnn(small_presence)

    y1 = np.full((X1.shape[0],), 0)
    y2 = np.full((X2.shape[0],), 1)
    y3 = np.full((X3.shape[0],), 2)

    X = np.vstack([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    print(" Class distribution:", Counter(y))
    X, y = shuffle(X, y, random_state=42)

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    return X, y

