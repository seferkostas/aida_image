import numpy as np
from art.attacks.evasion import (
    FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent,
    DeepFool, CarliniLInfMethod, BoundaryAttack, HopSkipJump, SquareAttack
)
from art.estimators.classification import TensorFlowV2Classifier
from .base_attack import BaseAttack

class AdversarialAttacks:
    def __init__(self, model, num_classes, input_shape, loss_object):
        self.classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=num_classes,
            input_shape=input_shape,
            loss_object=loss_object
        )
        
    def fgsm_attack(self, x_test, **params):
        """Fast Gradient Sign Method attack."""
        eps = params.get('eps', 0.3)
        norm = params.get('norm', 'inf')
        
        attack = FastGradientMethod(
            estimator=self.classifier,
            eps=eps,
            norm=norm
        )
        return attack.generate(x=x_test)
    
    def bim_attack(self, x_test, **params):
        """Basic Iterative Method attack."""
        eps = params.get('eps', 0.3)
        eps_step = params.get('eps_step', 0.01)
        max_iter = params.get('max_iter', 10)
        
        attack = BasicIterativeMethod(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )
        return attack.generate(x=x_test)
    
    def pgd_attack(self, x_test, **params):
        """Projected Gradient Descent attack."""
        eps = params.get('eps', 0.3)
        eps_step = params.get('eps_step', 0.01)
        max_iter = params.get('max_iter', 20)
        
        attack = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )
        return attack.generate(x=x_test)
    
    def deepfool_attack(self, x_test, **params):
        """DeepFool attack."""
        max_iter = params.get('max_iter', 50)
        epsilon = params.get('epsilon', 1e-6)
        
        attack = DeepFool(
            classifier=self.classifier,
            max_iter=max_iter,
            epsilon=epsilon
        )
        return attack.generate(x=x_test)
    
    def cw_attack(self, x_test, **params):
        """Carlini & Wagner attack."""
        confidence = params.get('confidence', 0.0)
        targeted = params.get('targeted', False)
        max_iter = params.get('max_iter', 10)
        
        attack = CarliniLInfMethod(
            classifier=self.classifier,
            confidence=confidence,
            targeted=targeted,
            max_iter=max_iter,
            learning_rate=params.get('learning_rate', 1e-2),
            initial_const=params.get('initial_const', 1e-3),
            batch_size=params.get('batch_size', 1),
            verbose=params.get('verbose', True)
        )
        return attack.generate(x=x_test)

    def boundary_attack(self, x_test, **params):
        """Boundary Attack (black-box)."""
        targeted = params.get('targeted', False)
        max_iter = params.get('max_iter', 50)   
        attack = BoundaryAttack(
            estimator=self.classifier,
            targeted=targeted,
            max_iter=max_iter,
            delta=params.get('delta', 0.01),
            epsilon=params.get('epsilon', 0.01),
            step_adapt=params.get('step_adapt', 0.90),
            init_size=params.get('init_size', 10),
            verbose=False
        )
        return attack.generate(x=x_test)

    def hopskipjump_attack(self, x_test, **params):
        """HopSkipJump Attack (black-box)."""
        targeted = params.get('targeted', False)
        max_iter = params.get('max_iter', 50)
        attack = HopSkipJump(
            classifier=self.classifier,
            targeted=targeted,
            max_iter=max_iter,
            max_eval=params.get('max_eval', 100),
            init_eval=params.get('init_eval', 10),
            init_size=params.get('init_size', 10),
            verbose=params.get('verbose', True)
        )
        return attack.generate(x=x_test)

    def square_attack(self, x_test, **params):
        """Square Attack (black-box)."""
        targeted = params.get('targeted', False)
        eps = params.get('eps', 0.3)
        max_iter = params.get('max_iter', 1000)
        attack = SquareAttack(
            estimator=self.classifier,
            targeted=targeted,
            eps=eps,
            max_iter=max_iter,
            p_init=params.get('p_init', 0.1)
        )
        return attack.generate(x=x_test)