import pandas as pd
import numpy as np
import functools
from sklearn.preprocessing import MinMaxScaler

def load_dataset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"--- Attempting to load dataset using '{func.__name__}' ---")
        dataset = func(*args, **kwargs)
        print(f"--- Successfully loaded dataset. Shape: {dataset.shape} ---")
        return dataset
    return wrapper

@load_dataset
def load_iris():
    file_path = 'data/Iris.csv'
    df = pd.read_csv(file_path)
    return df

def one_hot_encode(labels: np.array) -> np.array:
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    num_samples = len(labels)
    
    if num_samples == 0:
        return np.zeros((0, num_classes), dtype=int)

    class_to_index = {label: index for index, label in enumerate(unique_classes)}
    one_hot_encoded = np.zeros((num_samples, num_classes), dtype=int)

    for i, label in enumerate(labels):
        index = class_to_index[label]
        one_hot_encoded[i, index] = 1

    return one_hot_encoded

# Data Normalisation
# x_scaled = (x - min(data)) / (max(data) - min(data))
def minmax_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled