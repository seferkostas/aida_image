import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.experiment_config import (
    DATASETS, MODELS, ATTACKS, DEFENSES,
    TRAINING_DEFENSES, PREPROCESSING_DEFENSES, ExperimentConfig
)
from src.datasets.data_loader import DataLoader
from src.models.cnn_models import CNNModel, DeepCNN
from src.models.resnet_models import ResNetModel
from src.attacks.adversarial_attacks import AdversarialAttacks
from src.defenses.defense_methods import DefenseMethods
from src.evaluation.evaluator import AdversarialEvaluator
from src.utils.visualization import ResultsVisualizer

class ExperimentRunner:
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.results_dir = Path(experiment_config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = []
        self.training_history_records = []  # per epoch per dataset/model
        self.verbose = getattr(self.config, 'verbose', True)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Runner] {msg}")

    def run_full_experiment_suite(self):
        """Run complete experiment suite across all configurations."""
        print(f"Starting experiment suite: {self.config.name}")
        print(f"Datasets: {self.config.datasets}")
        print(f"Models: {self.config.models}")
        print(f"Attacks: {self.config.attacks}")
        print(f"Defenses: {self.config.defenses}")
        start_time = datetime.now()
        for dataset_name in self.config.datasets:
            for model_name in self.config.models:
                self._run_single_experiment(dataset_name, model_name)
        end_time = datetime.now()
        print(f"\nExperiment suite completed in {end_time - start_time}")
        self._save_and_visualize_results()

    def _run_single_experiment(self, dataset_name: str, model_name: str):
        self._log(f"Running {dataset_name} + {model_name}")


        # Load dataset
        dataset_config = DATASETS[dataset_name]
        data_loader = DataLoader(dataset_config)
        (x_train, y_train), (x_test, y_test) = data_loader.load_data()
        self._log(f"Dataset loaded: train={x_train.shape[0]}, test={x_test.shape[0]}")

        # Base model training
        self._log(f"Building model: {model_name}")
        model_config = MODELS[model_name]
        model_config.input_shape = dataset_config.input_shape
        model_config.num_classes = dataset_config.num_classes
        model = self._create_model(model_config)
        self._log("Starting base model training")
        model = self._train_model(model, x_train, y_train, x_test, y_test, dataset_name, model_name)
        self._log("Base model training completed")

        # Training defenses
        selected_training = [d for d in self.config.defenses if d in TRAINING_DEFENSES]
        selected_preproc = [d for d in self.config.defenses if d in PREPROCESSING_DEFENSES]

        base_defense_manager = DefenseMethods(
            model=model,
            num_classes=dataset_config.num_classes,
            input_shape=dataset_config.input_shape
        )
        model_variants = {'none': model}
        for train_def in selected_training:
            self._log(f"Training defense init: {train_def}")
            defense_cfg = DEFENSES[train_def]
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.build(model.input_shape)
            cloned_model.set_weights(model.get_weights())
            cloned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            train_fn = getattr(base_defense_manager, f"{train_def}_defense")
            trained_variant = train_fn(cloned_model, x_train, y_train, **defense_cfg.params)
            model_variants[train_def] = trained_variant
            self._log(f"Training defense completed: {train_def}")

        # Pre-processing defenses
        preprocessing_defenses = {'none': None}
        if selected_preproc:
            pre_manager = DefenseMethods(
                model=model,
                num_classes=dataset_config.num_classes,
                input_shape=dataset_config.input_shape
            )
            for pre_def in selected_preproc:
                self._log(f"Preprocess defense init: {pre_def}")
                dcfg = DEFENSES[pre_def]
                factory = getattr(pre_manager, f"{pre_def}_defense")
                obj = factory(**dcfg.params)

                def wrap(obj_ref, name=pre_def):
                    def preprocess(x):
                        out = obj_ref(x)
                        return out[0] if isinstance(out, tuple) else out
                    return type('WrappedDefense', (object,), {'name': name, 'preprocess': staticmethod(preprocess)})

                preprocessing_defenses[pre_def] = wrap(obj)
                self._log(f"Preprocess defense ready: {pre_def}")

        attack_names = ['none'] + self.config.attacks
        evaluator = AdversarialEvaluator(self.config)
        results = {'dataset': dataset_name, 'model': model_name, 'evaluations': []}
        self._log(f"Starting evaluations: train_variants={len(model_variants)}, preprocess={len(preprocessing_defenses)}, attacks={len(attack_names)}")

        for train_variant_name, train_model in model_variants.items():
            for pre_name, pre_obj in preprocessing_defenses.items():
                for attack_name in attack_names:
                    defense_label_parts = [p for p in [train_variant_name, pre_name] if p != 'none']
                    defense_label = '+'.join(defense_label_parts) if defense_label_parts else 'none'
                    self._log(f"Eval combo -> attack={attack_name}, defense={defense_label}")
                    if attack_name == 'none':
                        eval_result = evaluator._evaluate_attack_defense_combination(
                            train_model, None, pre_obj, x_test, y_test, 'none', defense_label
                        )
                        results['evaluations'].append(eval_result)
                        continue
                    attack_params = ATTACKS[attack_name].params if attack_name in ATTACKS else {}
                    attack_manager = AdversarialAttacks(
                        model=train_model,
                        num_classes=dataset_config.num_classes,
                        input_shape=dataset_config.input_shape,
                        loss_object=tf.keras.losses.CategoricalCrossentropy()
                    )
                    attack_fn = getattr(attack_manager, f"{attack_name}_attack")

                    def attack_callable(x, fn=attack_fn, params=attack_params):
                        return fn(x, **params)

                    eval_result = evaluator._evaluate_attack_defense_combination(
                        train_model, attack_callable, pre_obj, x_test, y_test, attack_name, defense_label
                    )
                    results['evaluations'].append(eval_result)

        for evaluation in results['evaluations']:
            evaluation['timestamp'] = datetime.now().isoformat()

        self.all_results.append(results)
        self._save_intermediate_results(results, dataset_name, model_name)
        self._log(f"Intermediate results saved for {dataset_name}+{model_name}")
        
    def _create_model(self, model_config):
        """Create model based on configuration."""
        
        if model_config.name.startswith('cnn'):
            return DeepCNN(
                model_config.input_shape,
                model_config.num_classes,
                model_config.architecture_params
            ).build_model()
        elif model_config.name.startswith('resnet'):
            return ResNetModel(
                model_config.input_shape,
                model_config.num_classes,
                model_config.architecture_params
            ).build_model()
        else:
            raise ValueError(f"Unknown model type: {model_config.name}")
            
    def _train_model(self, model, x_train, y_train, x_test, y_test, dataset_name: str, model_name: str):
        """Train model with standard configuration."""
        dataset_config = DATASETS[dataset_name]
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        # Record base training history (will merge with adversarial metrics if present)
        self._record_training_history(history, dataset_name, model_name)
        
        return model

    def _record_training_history(self, history: tf.keras.callbacks.History, dataset_name: str, model_name: str):
        """Flatten and store training history for later aggregation/visualization."""
        if not history or not hasattr(history, 'history'):
            return
        hist_dict = history.history
        # Determine number of epochs actually run
        epochs_run = len(next(iter(hist_dict.values()))) if hist_dict else 0
        for epoch_idx in range(epochs_run):
            metrics = {k: v[epoch_idx] for k, v in hist_dict.items() if k in ['loss','accuracy','val_loss','val_accuracy']}
            self._merge_epoch_metrics(dataset_name, model_name, epoch_idx + 1, metrics)

    def _merge_epoch_metrics(self, dataset_name: str, model_name: str, epoch: int, new_metrics: dict):
        """Merge metrics for (dataset, model, epoch) into training_history_records (create if missing)."""
        for rec in self.training_history_records:
            if rec['dataset']==dataset_name and rec['model']==model_name and rec['epoch']==epoch:
                rec.update({k:v for k,v in new_metrics.items() if v is not None})
                return
        base = {'dataset': dataset_name, 'model': model_name, 'epoch': epoch}
        base.update({k:v for k,v in new_metrics.items() if v is not None})
        self.training_history_records.append(base)
        
    def _save_intermediate_results(self, results, dataset_name, model_name):
        """Save intermediate results for backup."""
        
        filename = f"{dataset_name}_{model_name}.json"
        filepath = self.results_dir / 'raw_data' / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
    def _save_and_visualize_results(self):
        """Save final results and create visualizations."""
        
        # Compile all results into DataFrame
        evaluator = AdversarialEvaluator(self.config)
        evaluator.results = self.all_results
        results_df = evaluator.compile_results_dataframe()
        
        # Save DataFrame
        results_df.to_csv(self.results_dir / 'compiled_results.csv', index=False)

        # Save training history (if any)
        if self.training_history_records:
            training_df = pd.DataFrame(self.training_history_records)
            training_df.to_csv(self.results_dir / 'training_history.csv', index=False)
        else:
            training_df = pd.DataFrame()
        
        # Create visualizations
        visualizer = ResultsVisualizer(results_df, str(self.results_dir))
        visualizer.generate_all_visualizations()
        # Add training curves
        if not training_df.empty:
            visualizer.create_training_curves(training_df)
            visualizer.create_adversarial_accuracy_curves(training_df)
        
        # Save experiment configuration
        config_dict = {
            'name': self.config.name,
            'datasets': self.config.datasets,
            'models': self.config.models,
            'attacks': self.config.attacks,
            'defenses': self.config.defenses,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs
        }
        
        with open(self.results_dir / 'experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Compiled DataFrame shape: {results_df.shape}")
        print(f"Unique combinations tested: {len(results_df.groupby(['dataset', 'model', 'attack', 'defense']))}")