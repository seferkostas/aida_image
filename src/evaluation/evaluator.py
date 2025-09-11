import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from typing import Dict, List, Tuple, Any

class AdversarialEvaluator:
    def __init__(self, experiment_config):
        self.config = experiment_config
        self.results = []
        self.verbose = getattr(experiment_config, 'verbose', True)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Evaluator] {msg}")
        
    def evaluate_model_robustness(self, model, dataset_name, model_name, 
                                attacks, defenses, x_test, y_test) -> Dict[str, Any]:
        """Comprehensive evaluation of model robustness."""
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'evaluations': []
        }
        
        # Baseline evaluation (no attack, no defense)
        baseline_metrics = self._evaluate_predictions(model, x_test, y_test)
        baseline_metrics.update({
            'attack': 'none',
            'defense': 'none',
            'attack_params': {},
            'defense_params': {}
        })
        results['evaluations'].append(baseline_metrics)
        
        # Evaluate each attack-defense combination
        for attack_name, attack_obj in attacks.items():
            for defense_name, defense_obj in defenses.items():
                eval_result = self._evaluate_attack_defense_combination(
                    model, attack_obj, defense_obj, x_test, y_test,
                    attack_name, defense_name
                )
                results['evaluations'].append(eval_result)
                
        return results
    

    def _evaluate_attack_defense_combination(self, model, attack, defense, 
                                        x_test, y_test, attack_name, defense_name):
        """Evaluate specific attack-defense combination."""
        start_time = time.time()
        original_count = x_test.shape[0]
        self._log(f"Start eval: attack={attack_name}, defense={defense_name}, n={original_count}")
        
        # Apply defense preprocessing if applicable
        if defense and hasattr(defense, 'preprocess'):
            self._log(f"Preprocess start: {defense_name}")
            x_test_processed = defense.preprocess(x_test)
            self._log(f"Preprocess end: {defense_name} -> shape {getattr(x_test_processed, 'shape', '?')}")
        else:
            x_test_processed = x_test
        
        # Safeguard: some preprocessors (e.g., GaussianAugmentation) may augment and return more samples
        if x_test_processed.shape[0] != y_test.shape[0]:
            self._log(f"Alignment notice after preprocess: x={x_test_processed.shape[0]}, y={y_test.shape[0]} -> realigning")
            if x_test_processed.shape[0] > y_test.shape[0]:
                # Truncate augmented samples to keep alignment
                x_test_processed = x_test_processed[:y_test.shape[0]]
            else:
                # Truncate labels if somehow fewer samples are returned
                y_test = y_test[:x_test_processed.shape[0]]
        
        # Generate adversarial examples
        if attack:
            self._log(f"Attack generation start: {attack_name}")
            # Check if attack is a function (your attack methods) or an object with generate method
            if callable(attack):
                # Call the attack function directly (your implementation)
                x_adv = attack(x_test_processed)
            else:
                # Use generate method (ART-style attacks)
                x_adv = attack.generate(x_test_processed)
            self._log(f"Attack generation end: {attack_name} -> shape {getattr(x_adv, 'shape', '?')}")
        else:
            x_adv = x_test_processed
            self._log("No attack: using clean inputs")
        
        # If attack generation also altered sample count unexpectedly, realign
        if x_adv.shape[0] != y_test.shape[0]:
            self._log(f"Alignment notice after attack: x={x_adv.shape[0]}, y={y_test.shape[0]} -> realigning")
            if x_adv.shape[0] > y_test.shape[0]:
                x_adv = x_adv[:y_test.shape[0]]
            else:
                y_test = y_test[:x_adv.shape[0]]
        
        # Apply defense postprocessing if applicable
        if defense and hasattr(defense, 'postprocess'):
            self._log(f"Postprocess start: {defense_name}")
            x_final = defense.postprocess(x_adv)
            self._log(f"Postprocess end: {defense_name} -> shape {getattr(x_final, 'shape', '?')}")
        else:
            x_final = x_adv
        
        # Final alignment check before evaluation
        if x_final.shape[0] != y_test.shape[0]:
            self._log(f"Final alignment notice pre-eval: x={x_final.shape[0]}, y={y_test.shape[0]} -> truncating to match")
            min_len = min(x_final.shape[0], y_test.shape[0])
            x_final = x_final[:min_len]
            y_test = y_test[:min_len]
        
        # Evaluate
        self._log("Evaluating predictions...")
        metrics = self._evaluate_predictions(model, x_final, y_test)
        
        # Calculate perturbation metrics
        if attack:
            perturbation_metrics = self._calculate_perturbation_metrics(x_test_processed[:x_final.shape[0]], x_adv[:x_final.shape[0]])
        else:
            perturbation_metrics = {'l2_norm': 0.0, 'l_inf_norm': 0.0, 'success_rate': 0.0}
        
        eval_time = time.time() - start_time
        metrics.update({
            'attack': attack_name,
            'defense': defense_name,
            'evaluation_time': eval_time,
            **perturbation_metrics
        })
        self._log(f"Done eval: attack={attack_name}, defense={defense_name}, acc={metrics['accuracy']:.4f}, l2={metrics['l2_norm']:.4f}, time={eval_time:.2f}s")
        
        return metrics
    
    def _evaluate_predictions(self, model, x_test, y_test):
        """Calculate comprehensive prediction metrics."""
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        # Add zero_division=0 to avoid UndefinedMetricWarning when some classes are never predicted
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate confidence statistics
        confidence_scores = np.max(predictions, axis=1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }
    
    def _calculate_perturbation_metrics(self, x_original, x_adversarial):
        """Calculate perturbation strength metrics."""
        diff = x_adversarial - x_original
        
        # L2 norm (average per sample)
        l2_norms = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
        l2_norm = np.mean(l2_norms)
        
        # L-infinity norm (average per sample)
        l_inf_norms = np.max(np.abs(diff).reshape(len(diff), -1), axis=1)
        l_inf_norm = np.mean(l_inf_norms)
        
        # Success rate (how many samples were actually perturbed)
        success_rate = np.mean(l2_norms > 1e-8)
        
        return {
            'l2_norm': l2_norm,
            'l_inf_norm': l_inf_norm,
            'success_rate': success_rate
        }
    
    def compile_results_dataframe(self) -> pd.DataFrame:
        """Compile all results into a comprehensive DataFrame."""
        all_evaluations = []
        
        for result in self.results:
            dataset = result['dataset']
            model = result['model']
            
            for evaluation in result['evaluations']:
                evaluation.update({
                    'dataset': dataset,
                    'model': model
                })
                all_evaluations.append(evaluation)
                
        return pd.DataFrame(all_evaluations)