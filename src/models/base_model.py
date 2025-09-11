from abc import ABC, abstractmethod
import tensorflow as tf

class BaseModel(ABC):
    """Abstract base class for all model architectures."""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    @abstractmethod
    def build_model(self):
        """Build and return the model architecture."""
        pass
    
    def compile_model(self, model, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """Compile model with standard configuration."""
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def get_model_summary(self, model):
        """Get model summary information."""
        return {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': len(model.layers),
            'input_shape': self.input_shape,
            'output_shape': (None, self.num_classes)
        }