import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    name: str
    input_shape: tuple
    num_classes: int
    architecture_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DefenseConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    name: str
    input_shape: tuple
    num_classes: int
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    name: str
    datasets: List[str]
    models: List[str]
    attacks: List[str]
    defenses: List[str]
    metrics: List[str]
    batch_size: int = 64
    epochs: int = 10
    output_dir: str = "results"
    verbose: bool = True

# Predefined configurations
DATASETS = {
    'mnist': DatasetConfig(
        name='mnist',
        input_shape=(28, 28, 1),
        num_classes=10,
        preprocessing_params={'normalize': True, 'scale': 255.0}
    ),
    'fashion_mnist': DatasetConfig(
        name='fashion_mnist',
        input_shape=(28, 28, 1),
        num_classes=10,
        preprocessing_params={'normalize': True, 'scale': 255.0}
    ),
    'cifar10': DatasetConfig(
        name='cifar10',
        input_shape=(32, 32, 3),
        num_classes=10,
        preprocessing_params={'normalize': True, 'scale': 255.0}
    ),
    'cifar100': DatasetConfig(
        name='cifar100',
        input_shape=(32, 32, 3),
        num_classes=100,
        preprocessing_params={'normalize': True, 'scale': 255.0}
    )
}

MODELS = {
    'cnn': ModelConfig(
        name='cnn',
        input_shape=None,
        num_classes=None,
        architecture_params={'conv_layers': [32, 64, 128], 'dense_layers': [256, 128]}
    ),
    'resnet': ModelConfig(
        name='resnet',
        input_shape=None,
        num_classes=None,
        architecture_params={'num_blocks': 4, 'filters': [32, 64, 128, 256]}
    )
}

ATTACKS = {
    'fgsm': AttackConfig(
        name='fgsm',
        params={'eps': 0.1, 'norm': 'inf'}
    ),
    'bim': AttackConfig(
        name='bim',
        params={'eps': 0.1, 'eps_step': 0.01, 'max_iter': 3}
    ),
    'pgd': AttackConfig(
        name='pgd',
        params={'eps': 0.1, 'eps_step': 0.02, 'max_iter': 3}
    ),
    'deepfool': AttackConfig(
        name='deepfool',
        params={
            'max_iter': 3,         
            'epsilon': 1e-3,        
            'nb_grads': 3          
            # 'verbose': False
        }
    ),
    'cw': AttackConfig(
        name='cw',
        params={
            'confidence': 0.0,      
            'targeted': False,      
            'max_iter': 3,            
            'learning_rate': 1e-2,  
            'initial_const': 1e-3,
            'batch_size': 1,
            'verbose': True
        }
    ),
    'boundary': AttackConfig(
        name='boundary',
        params={
            'targeted': False,      
            'max_iter': 3,          
            'delta': 0.05,          
            'epsilon': 0.01,        
            'step_adapt': 0.90,     
            'init_size': 10,  
            'verbose': False
        }
    ),
    'hopskipjump': AttackConfig(
        name='hopskipjump',
        params={
            'targeted': False,      
            'max_iter': 3,         
            'max_eval': 75,       
            'init_eval': 10,       
            'init_size': 10,
            'verbose': True
        }
    ),
    'square': AttackConfig(
        name='square',
        params={
            'targeted': False,
            'max_iter': 3,
            'eps': 0.05,
            'p_init': 0.8
        }
    )
}

DEFENSES = {
    'none': DefenseConfig(name='none', params={}),
    'adversarial_training': DefenseConfig(
        name='adversarial_training',
    params={'attack_type': 'fgsm', 
            'ratio': 0.5, 
            'epochs': 1, 
            'eps': 0.1, 
            'batch_size': 64, 
            'memory_efficient': True, 
            'sample_fraction': 1.0}
    ),
    'feature_squeezing': DefenseConfig(
        name='feature_squeezing',
        params={'bit_depth': 2, 
                'clip_values': (0, 1)}
    ),
    'spatial_smoothing': DefenseConfig(
        
        name='spatial_smoothing',
        params={'window_size': 2, 
                'clip_values': (0, 1)}
    ),
    'adversarial_training_pgd': DefenseConfig(
        name='adversarial_training_pgd',
        params={'ratio': 0.5, 
            'epochs': 10, 
            'eps': 0.1, 
            'eps_step': 0.02, 
            'max_iter': 3, 
            'batch_size': 64, 
            'memory_efficient': True, 
            'sample_fraction': 1.0}
    ),
    'jpeg_compression': DefenseConfig(
        name='jpeg_compression',
        params={'quality': 80, 
                'clip_values': (0, 1)}
    ),
    'gaussian_augmentation': DefenseConfig(
        name='gaussian_augmentation',
        params={'sigma': 0.1, 
                'clip_values': (0, 1)}
    )
}

TRAINING_DEFENSES = {
    'adversarial_training',
    'adversarial_training_pgd'
}

PREPROCESSING_DEFENSES = {
    'feature_squeezing',
    'spatial_smoothing',
    'jpeg_compression',
    'gaussian_augmentation'
}