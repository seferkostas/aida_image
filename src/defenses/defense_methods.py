import numpy as np
import tensorflow as tf
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing, JpegCompression, GaussianAugmentation
from art.defences.postprocessor import GaussianNoise
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from .base_defense import BaseDefense

class DefenseMethods:
    def __init__(self, model, num_classes, input_shape):
        self.model = model
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.classifier = None
        self._setup_classifier()
        
    def _setup_classifier(self):
        """Setup ART classifier for defense methods."""
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.classifier = TensorFlowV2Classifier(
            model=self.model,
            nb_classes=self.num_classes,
            input_shape=self.input_shape,
            loss_object=loss_object
        )
    
    def adversarial_training_defense(self, model, x_train, y_train, **params):
        """Return a new model adversarially trained with FGSM (standard adversarial training).

        This does NOT return a pre/post-processing object; it produces a model variant.
        """
        eps = params.get('eps', 0.1)
        ratio = params.get('ratio', 0.5)
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 64)
        memory_efficient = params.get('memory_efficient', False)
        sample_fraction = params.get('sample_fraction', 1.0)

        # Optional dataset subsampling for memory/time reduction
        if sample_fraction < 1.0:
            n = int(len(x_train) * sample_fraction)
            idx = np.random.choice(len(x_train), n, replace=False)
            x_train_eff = x_train[idx]
            y_train_eff = y_train[idx]
        else:
            x_train_eff = x_train
            y_train_eff = y_train

        # Ensure model is compiled
        if not getattr(model, 'optimizer', None):
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = model.optimizer if getattr(model, 'optimizer', None) else tf.keras.optimizers.Adam()

        if not memory_efficient:
            # Use ART's trainer (may allocate large intermediate tensors)
            classifier = TensorFlowV2Classifier(
                model=model,
                nb_classes=self.num_classes,
                input_shape=self.input_shape,
                loss_object=loss_object,
                optimizer=optimizer,
                clip_values=(0.0, 1.0)
            )
            attack = FastGradientMethod(estimator=classifier, eps=eps)
            trainer = AdversarialTrainer(classifier, attacks=attack, ratio=ratio)
            trainer.fit(x_train_eff, y_train_eff, batch_size=batch_size, nb_epochs=epochs)
            return model

        # Memory-efficient custom FGSM adversarial training loop 
        train_size = x_train_eff.shape[0]
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        for epoch in range(epochs):
            # Shuffle indices each epoch
            indices = np.random.permutation(train_size)
            x_train_eff = x_train_eff[indices]
            y_train_eff = y_train_eff[indices]
            epoch_loss = []
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(start + batch_size, train_size)
                x_batch = x_train_eff[start:end]
                y_batch = y_train_eff[start:end]
                x_tensor = tf.convert_to_tensor(x_batch)
                y_tensor = tf.convert_to_tensor(y_batch)
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    preds = model(x_tensor, training=True)
                    loss = loss_object(y_tensor, preds)
                grad = tape.gradient(loss, x_tensor)
                # Generate FGSM adversarial examples for full batch
                adv_x = x_tensor + eps * tf.sign(grad)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                # Mix clean and adversarial examples according to ratio
                if ratio >= 1.0:
                    mix_x = adv_x
                    mix_y = y_tensor
                elif ratio <= 0.0:
                    mix_x = x_tensor
                    mix_y = y_tensor
                else:
                    k = int(ratio * x_tensor.shape[0])
                    if k == 0:
                        k = x_tensor.shape[0] // 2
                    mix_x = tf.concat([adv_x[:k], x_tensor[k:]], axis=0)
                    mix_y = tf.concat([y_tensor[:k], y_tensor[k:]], axis=0)
                batch_loss = model.train_on_batch(mix_x, mix_y, return_dict=True)
                epoch_loss.append(batch_loss['loss'] if isinstance(batch_loss, dict) else batch_loss)
            print(f"[FGSM AdvTrain][Epoch {epoch+1}/{epochs}] mean_loss={np.mean(epoch_loss):.4f}")
        return model
    
    def feature_squeezing_defense(self, **params):
        """Create feature squeezing preprocessor."""
        bit_depth = params.get('bit_depth', 1)
        clip_values = params.get('clip_values', (0, 1))
        
        return FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth)
    
    def spatial_smoothing_defense(self, **params):
        """Create spatial smoothing preprocessor."""
        window_size = params.get('window_size', 3)
        clip_values = params.get('clip_values', (0, 1))

        return SpatialSmoothing(window_size=window_size, clip_values=clip_values)

    def gaussian_noise_defense(self, **params):
        """Create Gaussian noise preprocessor."""
        sigma = params.get('sigma', 0.1)
        clip_values = params.get('clip_values', (0, 1))
        return GaussianNoise(sigma=sigma, clip_values=clip_values)
    
    def adversarial_training_pgd_defense(self, model, x_train, y_train, **params):
        """Return a new model adversarially trained with PGD."""
        eps = params.get('eps', 0.1)
        eps_step = params.get('eps_step', 0.02)
        max_iter = params.get('max_iter', 3)
        ratio = params.get('ratio', 0.5)
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 64)
        memory_efficient = params.get('memory_efficient', False)
        sample_fraction = params.get('sample_fraction', 1.0)

        # Optional dataset subsampling
        if sample_fraction < 1.0:
            n = int(len(x_train) * sample_fraction)
            idx = np.random.choice(len(x_train), n, replace=False)
            x_train_eff = x_train[idx]
            y_train_eff = y_train[idx]
        else:
            x_train_eff = x_train
            y_train_eff = y_train

        if not getattr(model, 'optimizer', None):
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = model.optimizer if getattr(model, 'optimizer', None) else tf.keras.optimizers.Adam()

        if not memory_efficient:
            classifier = TensorFlowV2Classifier(
                model=model,
                nb_classes=self.num_classes,
                input_shape=self.input_shape,
                loss_object=loss_object,
                optimizer=optimizer,
                clip_values=(0.0, 1.0)
            )
            attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps_step, max_iter=max_iter)
            trainer = AdversarialTrainer(classifier, attacks=attack, ratio=ratio)
            trainer.fit(x_train_eff, y_train_eff, batch_size=batch_size, nb_epochs=epochs)
            return model

        # Memory-efficient PGD loop
        train_size = x_train_eff.shape[0]
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        for epoch in range(epochs):
            indices = np.random.permutation(train_size)
            x_train_eff = x_train_eff[indices]
            y_train_eff = y_train_eff[indices]
            epoch_loss = []
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(start + batch_size, train_size)
                x_batch = x_train_eff[start:end]
                y_batch = y_train_eff[start:end]
                x_adv = tf.convert_to_tensor(x_batch)
                y_tensor = tf.convert_to_tensor(y_batch)
                # Iterative PGD
                for _ in range(max_iter):
                    with tf.GradientTape() as tape:
                        tape.watch(x_adv)
                        preds = model(x_adv, training=True)
                        loss = loss_object(y_tensor, preds)
                    grad = tape.gradient(loss, x_adv)
                    x_adv = x_adv + eps_step * tf.sign(grad)
                    # Project to epsilon ball & clip
                    x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
                    perturbation = tf.clip_by_value(x_adv - x_batch, -eps, eps)
                    x_adv = tf.clip_by_value(x_batch + perturbation, 0.0, 1.0)
                # Mix clean + adversarial
                if ratio >= 1.0:
                    mix_x = x_adv
                    mix_y = y_tensor
                elif ratio <= 0.0:
                    mix_x = x_batch
                    mix_y = y_tensor
                else:
                    k = int(ratio * x_adv.shape[0])
                    if k == 0:
                        k = x_adv.shape[0] // 2
                    mix_x = tf.concat([x_adv[:k], x_batch[k:]], axis=0)
                    mix_y = tf.concat([y_tensor[:k], y_tensor[k:]], axis=0)
                batch_loss = model.train_on_batch(mix_x, mix_y, return_dict=True)
                epoch_loss.append(batch_loss['loss'] if isinstance(batch_loss, dict) else batch_loss)
            print(f"[PGD AdvTrain][Epoch {epoch+1}/{epochs}] mean_loss={np.mean(epoch_loss):.4f}")
        return model

    def jpeg_compression_defense(self, **params):
        """Create JPEG compression preprocessor."""
        quality = params.get('quality', 75)
        clip_values = params.get('clip_values', (0, 1))
        return JpegCompression(clip_values=clip_values, quality=quality)

    def gaussian_augmentation_defense(self, **params):
        """Create Gaussian augmentation preprocessor."""
        sigma = params.get('sigma', 0.1)
        clip_values = params.get('clip_values', (0, 1))
        return GaussianAugmentation(clip_values=clip_values, sigma=sigma)

class CustomDefense(BaseDefense):
    """Custom defense implementation."""
    def __init__(self, defense_type, **kwargs):
        super().__init__(**kwargs)
        self.defense_type = defense_type
        
    def apply(self, x, **kwargs):
        """Apply the specified defense."""
        pass
    