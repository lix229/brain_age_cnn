import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse


def load_cv_results(results_dir):
    """Load cross-validation results"""
    # Load main results
    results_df = pd.read_csv(os.path.join(results_dir, 'grid_search_results.csv'))
    
    # Load best parameters
    with open(os.path.join(results_dir, 'best_parameters.json'), 'r') as f:
        best_params = json.load(f)
    
    return results_df, best_params


def create_visualizations(results_df, output_dir):
    """Create various visualizations of the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Learning rate vs MAE plot
    plt.figure(figsize=(10, 6))
    for batch_factor in results_df['batch_factor'].unique():
        df_subset = results_df[results_df['batch_factor'] == batch_factor]
        plt.errorbar(df_subset['learning_rate'], 
                    df_subset['avg_val_mae'],
                    yerr=df_subset['std_val_mae'],
                    marker='o', 
                    label=f'Batch factor: {batch_factor}',
                    capsize=5)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Validation MAE (years)')
    plt.title('Learning Rate vs Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_vs_mae.png'), dpi=300)
    plt.close()
    
    # 2. Heatmap of results
    if len(results_df['batch_factor'].unique()) > 1:
        pivot_df = results_df.pivot(index='batch_factor', 
                                   columns='learning_rate', 
                                   values='avg_val_mae')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlBu_r')
        plt.title('Validation MAE Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mae_heatmap.png'), dpi=300)
        plt.close()
    
    # 3. Training time comparison
    plt.figure(figsize=(10, 6))
    x = range(len(results_df))
    plt.bar(x, results_df['training_time'])
    plt.xlabel('Parameter Set ID')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time for Each Parameter Set')
    plt.xticks(x, results_df['param_id'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_times.png'), dpi=300)
    plt.close()
    
    # 4. Box plot of fold MAEs
    plt.figure(figsize=(12, 6))
    fold_data = []
    labels = []
    for idx, row in results_df.iterrows():
        fold_maes = eval(row['fold_maes'])  # Convert string to list
        fold_data.extend(fold_maes)
        labels.extend([f"Set {row['param_id']}"] * len(fold_maes))
    
    df_folds = pd.DataFrame({'MAE': fold_data, 'Parameter Set': labels})
    sns.boxplot(data=df_folds, x='Parameter Set', y='MAE')
    plt.xticks(rotation=45)
    plt.ylabel('Validation MAE (years)')
    plt.title('Distribution of Fold MAEs Across Parameter Sets')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_mae_distribution.png'), dpi=300)
    plt.close()


def create_summary_table(results_df, best_params, output_path):
    """Create a summary table of results"""
    # Sort by average MAE
    sorted_df = results_df.sort_values('avg_val_mae').reset_index(drop=True)
    
    # Create summary
    summary_df = sorted_df[['param_id', 'batch_size', 'learning_rate', 
                           'avg_val_mae', 'std_val_mae', 'training_time']].copy()
    
    # Format columns
    summary_df['learning_rate'] = summary_df['learning_rate'].apply(lambda x: f'{x:.6f}')
    summary_df['avg_val_mae'] = summary_df['avg_val_mae'].apply(lambda x: f'{x:.4f}')
    summary_df['std_val_mae'] = summary_df['std_val_mae'].apply(lambda x: f'{x:.4f}')
    summary_df['training_time'] = summary_df['training_time'].apply(lambda x: f'{x:.1f}s')
    
    # Rename columns
    summary_df.columns = ['ID', 'Batch Size', 'Learning Rate', 
                         'Avg MAE', 'Std MAE', 'Time']
    
    # Save as CSV and text
    summary_df.to_csv(output_path.replace('.txt', '.csv'), index=False)
    
    with open(output_path, 'w') as f:
        f.write("Cross-Validation Results Summary\n")
        f.write("="*60 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" + "="*60 + "\n")
        f.write("Best Parameters:\n")
        f.write(f"Batch size: {best_params['batch_size']}\n")
        f.write(f"Learning rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"Average MAE: {best_params['avg_val_mae']:.4f} ± {best_params['std_val_mae']:.4f}\n")


def main(args):
    # Load results
    results_df, best_params = load_cv_results(args.results_dir)
    
    print(f"Loaded {len(results_df)} parameter combinations")
    print(f"\nBest parameters:")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Learning rate: {best_params['learning_rate']:.6f}")
    print(f"  Average MAE: {best_params['avg_val_mae']:.4f} ± {best_params['std_val_mae']:.4f}")
    
    # Create output directory
    output_dir = os.path.join(args.results_dir, 'analysis')
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_visualizations(results_df, output_dir)
    
    # Create summary table
    print(f"Creating summary table...")
    create_summary_table(results_df, best_params, 
                        os.path.join(output_dir, 'results_summary.txt'))
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze CV grid search results')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory containing CV results')
    
    args = parser.parse_args()
    main(args)