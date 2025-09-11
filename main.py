"""
Adversarial Machine Learning Experiment Framework
================================================================

This framework provides a complete system for evaluating the robustness of deep learning
models against adversarial attacks and the effectiveness of various defense methods.

Usage:
    python main.py --datasets mnist cifar10 --models cnn resnet --attacks fgsm pgd --defenses none adversarial_training --epochs 10 --output-dir results/experiment-1
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.experiment_config import ExperimentConfig, DATASETS, MODELS, ATTACKS, DEFENSES
from experiments.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive adversarial Deep Learning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

    # Custom configuration
    python main.py --datasets mnist cifar10 --models cnn --attacks fgsm pgd --defenses none adversarial_training --epochs 2 --output-dir results/experiment-1
    
    # List available options
    python main.py --list-options
        """
    )
    
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=list(DATASETS.keys()),
        help='Datasets to use in experiments'
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=list(MODELS.keys()),
        help='Models to evaluate'
    )
    
    parser.add_argument(
        '--attacks', 
        nargs='+', 
        choices=list(ATTACKS.keys()),
        help='Attack methods to test'
    )
    
    parser.add_argument(
        '--defenses', 
        nargs='+', 
        choices=list(DEFENSES.keys()),
        help='Defense methods to evaluate'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--list-options', 
        action='store_true',
        help='List all available datasets, models, attacks, and defenses'
    )
    
    args = parser.parse_args()
    
    if args.list_options:
        print("Available options:")
        print(f"Datasets: {list(DATASETS.keys())}")
        print(f"Models: {list(MODELS.keys())}")
        print(f"Attacks: {list(ATTACKS.keys())}")
        print(f"Defenses: {list(DEFENSES.keys())}")
        return
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        name='custom_experiment',
        datasets=args.datasets,
        models=args.models,
        attacks=args.attacks,
        defenses=args.defenses,
        metrics=['accuracy', 'f1_score', 'precision', 'recall'],
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # Run experiments
    print("=" * 80)
    print("ADVERSARIAL ATTACKS ON DEEP LEARNING EXPERIMENTS")
    print("=" * 80)
    
    runner = ExperimentRunner(experiment_config)
    
    try:
        runner.run_full_experiment_suite()
        print("\n" + "=" * 80)
        print("Experiment run successfully!")
        print(f"Results saved to: {experiment_config.output_dir}")
        print("Check the 'plots' directory for visualizations and 'tables' for summary tables.")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()