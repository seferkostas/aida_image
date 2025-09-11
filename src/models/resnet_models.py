import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Add, Input, 
    GlobalAveragePooling2D, Dense, MaxPooling2D
)
from tensorflow.keras.models import Model
from .base_model import BaseModel

class ResNetModel(BaseModel):
    def __init__(self, input_shape, num_classes, architecture_params=None):
        super().__init__(input_shape, num_classes)
        self.architecture_params = architecture_params or {}
        
    def residual_block(self, x, filters, stride=1, conv_shortcut=False, block_name=""):
        """Create a residual block."""
        if conv_shortcut:
            shortcut = Conv2D(filters, 1, strides=stride, name=f'{block_name}_0_conv')(x)
            shortcut = BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{block_name}_0_bn')(shortcut)
        else:
            shortcut = x
            
        x = Conv2D(filters, 3, strides=stride, padding='same', name=f'{block_name}_1_conv')(x)
        x = BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{block_name}_1_bn')(x)
        x = ReLU(name=f'{block_name}_1_relu')(x)
        
        x = Conv2D(filters, 3, padding='same', name=f'{block_name}_2_conv')(x)
        x = BatchNormalization(axis=3, epsilon=1.001e-5, name=f'{block_name}_2_bn')(x)
        
        x = Add(name=f'{block_name}_add')([shortcut, x])
        x = ReLU(name=f'{block_name}_out')(x)
        return x
    
    def stack_residual_blocks(self, x, filters, blocks, stride1=2, name=None):
        """Stack multiple residual blocks."""
        x = self.residual_block(x, filters, stride=stride1, conv_shortcut=True, block_name=f'{name}_block1')
        for i in range(2, blocks + 1):
            x = self.residual_block(x, filters, conv_shortcut=False, block_name=f'{name}_block{i}')
        return x
    
    def build_model(self):
        """Build ResNet model."""
        num_blocks = self.architecture_params.get('num_blocks', 4)
        img_size = self.input_shape[0]
        # Dataset-aware default filters if not provided
        filters = self.architecture_params.get(
            'filters',
            [32, 64, 128] if img_size <= 32 else [64, 128, 256, 512]
        )
        
        inputs = Input(shape=self.input_shape)
        
        # Stem: small images use 3x3 s1, large images use 7x7 s2 + pool
        if img_size <= 32:
            x = Conv2D(min(64, filters[0]), 3, strides=1, padding='same', name='conv1_conv')(inputs)
            x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
            x = ReLU(name='conv1_relu')(x)
            # No initial pooling for small images
        else:
            x = Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
            x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
            x = ReLU(name='conv1_relu')(x)
            x = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)
        
        # Residual blocks
        for i, f in enumerate(filters[:num_blocks]):
            stride = 1 if i == 0 else 2
            x = self.stack_residual_blocks(x, f, 2, stride1=stride, name=f'conv{i+2}')
        
        # Head
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs, outputs, name='resnet')
        return model