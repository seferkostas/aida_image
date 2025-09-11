from abc import ABC, abstractmethod
import numpy as np

class BaseDefense(ABC):
    """Abstract base class for defense methods."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        
    @abstractmethod
    def apply(self, x, **kwargs):
        """Apply defense to input data."""
        pass
    
    def preprocess(self, x):
        """Preprocess data before model inference."""
        return self.apply(x)
    
    def postprocess(self, x):
        """Postprocess data after adversarial generation."""
        return x
    
    def _validate_input(self, x):
        """Validate input data."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be numpy array")
        return True