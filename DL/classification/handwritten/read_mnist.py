import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_idx_images(path):
    with open(path,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[16:].reshape(-1, 28*28) / 255.0

def load_idx_labels(path):
    with open(path,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[8:]

def load_mnist():
    X_full = load_idx_images('mnist/train-images.idx3-ubyte')
    y_full = load_idx_labels('mnist/train-labels.idx1-ubyte')
    X_test = load_idx_images('mnist/t10k-images.idx3-ubyte')
    y_test = load_idx_labels('mnist/t10k-labels.idx1-ubyte')

    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=10000, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


#X_full = load_idx_images('mnist/train-images.idx3-ubyte')
#y_full = load_idx_labels('mnist/train-labels.idx1-ubyte')
#X_test = load_idx_images('mnist/t10k-images.idx3-ubyte')
#y_test = load_idx_labels('mnist/t10k-labels.idx1-ubyte')
#
#X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=10000, random_state=42)
#
#
#im = X_train[10000, :].reshape(28,28)
#print(y_train[10000])
#plt.imshow(im)
#plt.show()