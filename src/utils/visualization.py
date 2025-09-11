import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, results_df: pd.DataFrame, output_dir: str = "results"):
        self.df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_accuracy_comparison_plots(self):
        """Create comprehensive accuracy comparison plots."""
        
        # 1. Heatmap of accuracy by dataset and attack
        pivot_accuracy = self.df.pivot_table(
            values='accuracy', 
            index=['dataset', 'model'], 
            columns=['attack', 'defense'],
            aggfunc='mean'
        )
        
        plt.figure(figsize=(20, 12))
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'})
        plt.title('Model Accuracy Under Different Attack-Defense Combinations')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot comparing defenses
        defense_comparison = self.df.groupby(['defense', 'attack'])['accuracy'].mean().reset_index()
        
        plt.figure(figsize=(15, 8))
        sns.barplot(data=defense_comparison, x='attack', y='accuracy', hue='defense')
        plt.title('Defense Effectiveness Across Different Attacks')
        plt.ylabel('Mean Accuracy')
        plt.xlabel('Attack Type')
        plt.xticks(rotation=45)
        plt.legend(title='Defense Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'defense_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model robustness comparison
        model_robustness = self.df[self.df['attack'] != 'none'].groupby(['model', 'dataset'])['accuracy'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df[self.df['attack'] != 'none'], x='model', y='accuracy', hue='dataset')
        plt.title('Model Robustness Across Datasets')
        plt.ylabel('Accuracy Under Attack')
        plt.xlabel('Model Architecture')
        plt.xticks(rotation=45)
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'model_robustness.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_perturbation_analysis_plots(self):
        """Create plots analyzing perturbation strengths."""
        
        attack_data = self.df[self.df['attack'] != 'none'].copy()
        
        # 1. L2 vs L-inf norm scatter plot
        plt.figure(figsize=(12, 8))
        for attack in attack_data['attack'].unique():
            subset = attack_data[attack_data['attack'] == attack]
            plt.scatter(subset['l2_norm'], subset['l_inf_norm'], 
                       label=attack, alpha=0.7, s=60)
        
        plt.xlabel('L2 Norm')
        plt.ylabel('L∞ Norm')
        plt.title('Perturbation Strength Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'perturbation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Success rate vs accuracy trade-off
        plt.figure(figsize=(12, 8))
        for attack in attack_data['attack'].unique():
            subset = attack_data[attack_data['attack'] == attack]
            plt.scatter(subset['success_rate'], subset['accuracy'], 
                       label=attack, alpha=0.7, s=60)
        
        plt.xlabel('Attack Success Rate')
        plt.ylabel('Model Accuracy')
        plt.title('Attack Success Rate vs Model Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'success_rate_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def export_tables(self):
        """Export summary tables with the results in csv format."""
        # Summary statistics table
        summary_stats = self.df.groupby(['dataset', 'model', 'attack', 'defense']).agg({
            'accuracy': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'l2_norm': 'mean',
            'l_inf_norm': 'mean'
        }).round(4)
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        # Export to CSV
        summary_stats.to_csv(self.output_dir / 'tables' / 'results_summary.csv', index=False)

        # Defense effectiveness ranking table
        defense_ranking = self.df[self.df['attack'] != 'none'].groupby('defense')['accuracy'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4).sort_values('mean', ascending=False)
        # Export to CSV
        defense_ranking.to_csv(self.output_dir / 'tables' / 'defense_ranking.csv')

        # Model robustness table
        model_robustness = self.df[self.df['attack'] != 'none'].pivot_table(
            values='accuracy',
            index='model',
            columns='dataset',
            aggfunc='mean'
        ).round(4)
        # Export to CSV
        model_robustness.to_csv(self.output_dir / 'tables' / 'model_robustness.csv')

        print(f"Tables exported to {self.output_dir / 'tables'} (CSV)")
        
    def generate_all_visualizations(self):
        """Generate all visualizations and exports."""
        
        # Create directories
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        
        print("Generating accuracy comparison plots...")
        self.create_accuracy_comparison_plots()
        
        print("Generating perturbation analysis plots...")
        self.create_perturbation_analysis_plots()

        print("Exporting summary tables...")
        self.export_tables()

        print(f"All results from experiment saved to {self.output_dir}")


    def create_training_curves(self, training_df: pd.DataFrame):
        """Plot training & validation accuracy/loss over epochs per dataset with per-model lines."""
        if training_df.empty:
            return
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Ensure expected columns exist
        required_cols = {'dataset', 'model', 'epoch'}
        if not required_cols.issubset(training_df.columns):
            return

        # Accuracy Curves
        for dataset in training_df['dataset'].unique():
            subset = training_df[training_df['dataset'] == dataset]
            plt.figure(figsize=(10, 6))
            for model in subset['model'].unique():
                model_hist = subset[subset['model'] == model]
                if 'accuracy' in model_hist.columns:
                    plt.plot(model_hist['epoch'], model_hist['accuracy'], marker='o', label=f"{model} train acc")
                if 'val_accuracy' in model_hist.columns:
                    plt.plot(model_hist['epoch'], model_hist['val_accuracy'], linestyle='--', marker='x', label=f"{model} val acc")
            plt.title(f'Training Accuracy Over Epochs ({dataset})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(fontsize='small', ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f'training_accuracy_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Loss Curves
        for dataset in training_df['dataset'].unique():
            subset = training_df[training_df['dataset'] == dataset]
            plt.figure(figsize=(10, 6))
            for model in subset['model'].unique():
                model_hist = subset[subset['model'] == model]
                if 'loss' in model_hist.columns:
                    plt.plot(model_hist['epoch'], model_hist['loss'], marker='o', label=f"{model} train loss")
                if 'val_loss' in model_hist.columns:
                    plt.plot(model_hist['epoch'], model_hist['val_loss'], linestyle='--', marker='x', label=f"{model} val loss")
            plt.title(f'Training Loss Over Epochs ({dataset})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(fontsize='small', ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f'training_loss_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.close()


    def create_adversarial_accuracy_curves(self, training_df: pd.DataFrame):
        """Plot validation accuracy on clean vs FGSM vs PGD adversarial sets across epochs."""
        required = {'dataset','model','epoch'}
        if not required.issubset(training_df.columns):
            return
        if not any(c in training_df.columns for c in ['adv_fgsm_accuracy','adv_pgd_accuracy']):
            return
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        for dataset in training_df['dataset'].unique():
            ds_subset = training_df[training_df['dataset']==dataset]
            for model in ds_subset['model'].unique():
                msub = ds_subset[ds_subset['model']==model].sort_values('epoch')
                plt.figure(figsize=(9,5))
                if 'val_accuracy_clean' in msub:
                    plt.plot(msub['epoch'], msub['val_accuracy_clean'], marker='o', label='Clean (val)')
                elif 'val_accuracy' in msub:
                    plt.plot(msub['epoch'], msub['val_accuracy'], marker='o', label='Clean (val)')
                if 'adv_fgsm_accuracy' in msub and msub['adv_fgsm_accuracy'].notna().any():
                    plt.plot(msub['epoch'], msub['adv_fgsm_accuracy'], marker='x', linestyle='--', label='FGSM')
                if 'adv_pgd_accuracy' in msub and msub['adv_pgd_accuracy'].notna().any():
                    plt.plot(msub['epoch'], msub['adv_pgd_accuracy'], marker='s', linestyle='-.', label='PGD')
                plt.title(f'Adversarial Validation Accuracy ({dataset} - {model})')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.ylim(0,1)
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f'adversarial_val_accuracy_{dataset}_{model}.png', dpi=300, bbox_inches='tight')
                plt.close()