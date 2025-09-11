import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np
from typing import Tuple, Optional
import requests
import os

class DataLoader:
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.name = dataset_config.name
        
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess data based on dataset configuration."""
        
        if self.name == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif self.name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        elif self.name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif self.name == 'cifar100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")
            
        # Preprocess data
        x_train, x_test = self._preprocess_data(x_train, x_test)
        y_train, y_test = self._preprocess_labels(y_train, y_test)
        
        return (x_train, y_train), (x_test, y_test)
    
    def _preprocess_data(self, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image data."""
        # Ensure correct shape
        if len(x_train.shape) == 3:  # Add channel dimension for grayscale
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
            
        # Normalize if specified
        if self.config.preprocessing_params.get('normalize', False):
            scale = self.config.preprocessing_params.get('scale', 255.0)
            x_train = x_train.astype('float32') / scale
            x_test = x_test.astype('float32') / scale
            
        return x_train, x_test
    
    def _preprocess_labels(self, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert labels to categorical."""
        y_train = to_categorical(y_train, self.config.num_classes)
        y_test = to_categorical(y_test, self.config.num_classes)
        return y_train, y_test