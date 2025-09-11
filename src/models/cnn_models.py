import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    Dense, Dropout, BatchNormalization, Input, Flatten
)
from .base_model import BaseModel

class CNNModel(BaseModel):
    def __init__(self, input_shape, num_classes, architecture_params=None):
        super().__init__(input_shape, num_classes)
        self.architecture_params = architecture_params or {}
        
    def build_model(self):
        """Build CNN model based on architecture parameters."""
        conv_layers = self.architecture_params.get('conv_layers', [32, 64])
        dense_layers = self.architecture_params.get('dense_layers', [128])
        dropout_rate = self.architecture_params.get('dropout_rate', 0.5)
        
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        # Convolutional layers
        for i, filters in enumerate(conv_layers):
            model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            if i % 2 == 1 or i == len(conv_layers) - 1:  # Pool every 2 layers or at the end
                model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(dropout_rate * 0.5))  # Lighter dropout in conv layers
        
        # Global pooling or flatten
        if len(self.input_shape) == 3:  # 2D images
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())
        
        # Dense layers
        for units in dense_layers:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model

class DeepCNN(BaseModel):
    def __init__(self, input_shape, num_classes, architecture_params=None):
        super().__init__(input_shape, num_classes)
        self.architecture_params = architecture_params or {}
        
    def build_model(self):
        """Build deeper CNN with more complex architecture."""
        conv_layers = self.architecture_params.get('conv_layers', [64, 128, 256, 512])
        dense_layers = self.architecture_params.get('dense_layers', [512, 256])
        
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional blocks
        for i, filters in enumerate(conv_layers):
            # Two conv layers per block
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.25)(x)
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers
        for units in dense_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.5)(x)
        
        # Output
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs)