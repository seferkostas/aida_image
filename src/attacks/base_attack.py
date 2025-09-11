from abc import ABC, abstractmethod
import numpy as np

class BaseAttack(ABC):
    """Abstract base class for adversarial attacks."""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.params = kwargs
        
    @abstractmethod
    def generate(self, x, y=None, **kwargs):
        """Generate adversarial examples."""
        pass
    
    def _check_inputs(self, x):
        """Validate input data."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be numpy array")
        if len(x.shape) != 4:
            raise ValueError("Input must be 4D array (batch_size, height, width, channels)")
        return True