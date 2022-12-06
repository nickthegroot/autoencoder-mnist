import numpy as np
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST


def train_pca(data_dir: str, n_components: int):
    dataset = MNIST(data_dir, train=True, download=True)
    pca = PCA(n_components=n_components)

    data = dataset.data.numpy()
    data = data.reshape(-1, 28 * 28)  # Need to reshape into flattened array for PCA
    pca.fit(data)

    return pca
