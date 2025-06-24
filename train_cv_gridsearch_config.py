"""
Grid search with cross-validation using configuration system
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from datetime import datetime
import argparse
import json
from itertools import product
import time
import sys

from config import ExperimentConfig
from data_loader_csv import load_data_from_csv


def create_finetuned_model(config, learning_rate):
    """Create model for finetuning based on configuration"""
    # Load base model
    base_model = keras.models.load_model(config.model.base_model_path, compile=False)
    
    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze specified layers
    for layer_name in config.model.finetune_layers:
        for layer in base_model.layers:
            if layer.name == layer_name:
                layer.trainable = True
    
    # Unfreeze VGG blocks if specified
    if not config.model.freeze_backbone and config.model.finetune_vgg_blocks > 0:
        vgg16_model = base_model.layers[0]
        n_layers_to_unfreeze = config.model.finetune_vgg_blocks * 4
        for layer in vgg16_model.layers[-n_layers_to_unfreeze:]:
            layer.trainable = True
    
    # Compile with specified learning rate
    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return base_model


def load_images_from_df(df, config):
    """Load images from dataframe"""
    images = []
    ages = []
    patient_ids = []
    
    for _, row in df.iterrows():
        # Load image
        img_path = os.path.join(config.data.image_base_dir, row[config.data.image_col])
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(config.data.image_size)
        img_array = np.array(img)
        
        if config.data.normalize:
            img_array = img_array / 255.0
        
        images.append(img_array)
        ages.append(row[config.data.age_col])
        
        if config.data.patient_id_col:
            patient_ids.append(row[config.data.patient_id_col])
    
    return np.array(images), np.array(ages), patient_ids


def train_fold(X_train, y_train, X_val, y_val, config,
               batch_size, learning_rate, fold_num, output_dir, total_folds=3):
    """Train a single fold"""
    print(f"\n  Fold {fold_num}/{total_folds}: Training...")
    
    # Create model
    model = create_finetuned_model(config, learning_rate)
    
    # Create callbacks
    fold_dir = os.path.join(output_dir, f'fold_{fold_num}')
    os.makedirs(fold_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(fold_dir, 'best_model.h5'),
            monitor='val_mae',
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.training.reduce_lr_factor,
            patience=config.training.reduce_lr_patience,
            min_lr=config.training.min_lr,
            verbose=0
        )
    ]
    
    # Create progress callback
    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self, epochs, fold_num, total_folds):
            self.epochs = epochs
            self.fold_num = fold_num
            self.total_folds = total_folds
            self.start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self.start_time
            eta = (elapsed / (epoch + 1)) * (self.epochs - epoch - 1)
            progress = (epoch + 1) / self.epochs * 100
            
            print(f"\r  Fold {self.fold_num}/{self.total_folds}: Epoch {epoch+1}/{self.epochs} "
                  f"[{'='*int(progress/5):20s}] {progress:.1f}% - "
                  f"loss: {logs.get('loss', 0):.4f} - mae: {logs.get('mae', 0):.4f} - "
                  f"val_loss: {logs.get('val_loss', 0):.4f} - val_mae: {logs.get('val_mae', 0):.4f} - "
                  f"ETA: {int(eta)}s", end='', flush=True)
    
    progress_callback = ProgressCallback(config.training.epochs, fold_num, total_folds)
    callbacks.append(progress_callback)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=config.training.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0
    )
    print()  # New line after training
    
    # Get best validation MAE
    best_val_mae = min(history.history['val_mae'])
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(fold_dir, 'history.csv'), index=False)
    
    return best_val_mae, model


def perform_grid_search_cv(config):
    """Perform grid search with cross-validation using configuration"""
    
    # Validate configuration
    config.validate()
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_data_from_csv(config)
    
    # Combine train and validation for CV (we'll split them differently)
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Load all images
    images, ages, patient_ids = load_images_from_df(full_train_df, config)
    
    print(f"Loaded {len(images)} images for cross-validation")
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f} years")
    
    # Define hyperparameter grid from config
    batch_factors = config.grid_search.batch_factors
    loss_functions = config.grid_search.loss_functions
    learning_rates = config.grid_search.learning_rates
    
    # Base batch size
    base_batch_size = config.training.batch_size
    
    print(f"\nHyperparameter Grid:")
    print(f"Batch factors: {batch_factors}")
    print(f"Loss functions: {loss_functions}")
    print(f"Learning rates: {learning_rates}")
    print(f"Total combinations: {len(batch_factors) * len(loss_functions) * len(learning_rates)}")
    
    # Create output directory
    exp_dir = config.get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(exp_dir, 'config.json'))
    
    # Initialize results storage
    results = []
    
    # Create KFold object
    kf = KFold(n_splits=config.grid_search.n_folds, 
               shuffle=config.grid_search.shuffle_folds, 
               random_state=config.data.random_seed)
    
    # Grid search with progress tracking
    total_combinations = len(batch_factors) * len(loss_functions) * len(learning_rates)
    param_id = 0
    
    print(f"\n{'='*70}")
    print(f"STARTING GRID SEARCH")
    print(f"Total parameter combinations to test: {total_combinations}")
    print(f"Cross-validation folds: {config.grid_search.n_folds}")
    print(f"Total models to train: {total_combinations * config.grid_search.n_folds}")
    print(f"{'='*70}")
    
    overall_start_time = time.time()
    
    for batch_factor, loss_fn, lr in product(batch_factors, loss_functions, learning_rates):
        param_id += 1
        batch_size = int(base_batch_size * batch_factor)
        
        # Calculate time estimates
        elapsed_time = time.time() - overall_start_time
        if param_id > 1:
            avg_time_per_param = elapsed_time / (param_id - 1)
            remaining_params = total_combinations - param_id + 1
            eta_seconds = avg_time_per_param * remaining_params
            eta_str = f"{int(eta_seconds/3600)}h {int((eta_seconds%3600)/60)}m {int(eta_seconds%60)}s"
        else:
            eta_str = "Calculating..."
            
        print(f"\n{'='*70}")
        print(f"PARAMETER SET {param_id}/{total_combinations}")
        print(f"Progress: [{('█' * int((param_id-1)/total_combinations*20)):20s}] {(param_id-1)/total_combinations*100:.1f}%")
        print(f"Elapsed: {int(elapsed_time/3600)}h {int((elapsed_time%3600)/60)}m {int(elapsed_time%60)}s | ETA: {eta_str}")
        print(f"{'='*70}")
        print(f"Batch size: {batch_size} (factor: {batch_factor})")
        print(f"Loss function: {loss_fn}")
        print(f"Learning rate: {lr:.6f}")
        print(f"{'='*70}")
        
        # Create directory for this parameter combination
        param_dir = os.path.join(exp_dir, f'params_{param_id}')
        os.makedirs(param_dir, exist_ok=True)
        
        # Store parameter info
        param_info = {
            'param_id': param_id,
            'batch_size': batch_size,
            'batch_factor': batch_factor,
            'loss_function': loss_fn,
            'learning_rate': lr
        }
        
        with open(os.path.join(param_dir, 'parameters.json'), 'w') as f:
            json.dump(param_info, f, indent=2)
        
        # Cross-validation
        fold_maes = []
        start_time = time.time()
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(images), 1):
            fold_start_time = time.time()
            print(f"\n  Starting Fold {fold_num}/{config.grid_search.n_folds}")
            
            # Split data
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = ages[train_idx], ages[val_idx]
            
            # Train fold
            fold_mae, _ = train_fold(
                X_train, y_train, X_val, y_val,
                config, batch_size, lr,
                fold_num, param_dir, config.grid_search.n_folds
            )
            
            fold_maes.append(fold_mae)
            fold_time = time.time() - fold_start_time
            print(f"\n  Fold {fold_num} completed - Best VAL MAE: {fold_mae:.4f} - Time: {fold_time:.1f}s")
            
            # Estimate remaining time
            avg_fold_time = (time.time() - start_time) / fold_num
            remaining_folds = config.grid_search.n_folds - fold_num
            remaining_params = total_combinations - param_id
            total_remaining_folds = remaining_folds + remaining_params * config.grid_search.n_folds
            eta = avg_fold_time * total_remaining_folds
            
            print(f"  Estimated time remaining for grid search: {int(eta/60)}m {int(eta%60)}s")
        
        # Calculate average performance
        avg_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        training_time = time.time() - start_time
        
        print(f"\nAverage VAL MAE: {avg_mae:.4f} ± {std_mae:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Store results
        result = {
            'param_id': param_id,
            'batch_size': batch_size,
            'batch_factor': batch_factor,
            'loss_function': loss_fn,
            'learning_rate': lr,
            'avg_val_mae': avg_mae,
            'std_val_mae': std_mae,
            'fold_maes': fold_maes,
            'training_time': training_time
        }
        results.append(result)
        
        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(exp_dir, 'grid_search_results.csv'), index=False)
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['avg_val_mae'].idxmin()
    best_params = results_df.iloc[best_idx]
    
    total_time = time.time() - overall_start_time
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETED!")
    print(f"Total time: {int(total_time/3600)}h {int((total_time%3600)/60)}m {int(total_time%60)}s")
    print(f"Average time per parameter set: {total_time/total_combinations:.1f}s")
    print(f"{'='*70}")
    print(f"\nBEST PARAMETERS:")
    print(f"Batch size: {best_params['batch_size']} (factor: {best_params['batch_factor']})")
    print(f"Loss function: {best_params['loss_function']}")
    print(f"Learning rate: {best_params['learning_rate']:.6f}")
    print(f"Average VAL MAE: {best_params['avg_val_mae']:.4f} ± {best_params['std_val_mae']:.4f}")
    print(f"{'='*70}")
    
    # Save final results
    with open(os.path.join(exp_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params.to_dict(), f, indent=2)
    
    # Create and save best configuration for future reuse
    best_config = create_best_config(config, best_params)
    best_config.save(os.path.join(exp_dir, 'best_config.json'))
    best_config.save(os.path.join(exp_dir, 'best_config.yaml'))
    print(f"\nBest configuration saved to:")
    print(f"  - {os.path.join(exp_dir, 'best_config.json')}")
    print(f"  - {os.path.join(exp_dir, 'best_config.yaml')}")
    
    # Create summary report
    create_summary_report(config, results_df, best_params, exp_dir)
    
    print(f"\nResults saved to: {exp_dir}")
    
    return results_df, best_params


def create_best_config(original_config, best_params):
    """
    Create a new configuration object with the best hyperparameters found during grid search.
    
    Args:
        original_config: Original ExperimentConfig used for grid search
        best_params: Dictionary/Series containing best parameters from grid search
        
    Returns:
        ExperimentConfig: New configuration with optimal hyperparameters
    """
    from copy import deepcopy
    
    # Create a deep copy of the original configuration
    best_config = deepcopy(original_config)
    
    # Update configuration with best parameters
    best_config.name = f"{original_config.name}_best"
    best_config.description = (f"Best configuration from grid search. "
                              f"Original: {original_config.description}")
    
    # Update training parameters with best values
    best_config.training.batch_size = int(best_params['batch_size'])
    best_config.training.learning_rate = float(best_params['learning_rate'])
    
    # Reset grid search settings since this is for single training
    best_config.grid_search.n_folds = 3  # Keep for reference but not used in standard training
    best_config.grid_search.batch_factors = [1.0]  # Single value
    best_config.grid_search.learning_rates = [float(best_params['learning_rate'])]
    
    # Add metadata about the grid search
    best_config.description += (f" | Best MAE: {best_params['avg_val_mae']:.4f} ± "
                               f"{best_params['std_val_mae']:.4f}")
    
    return best_config


def create_summary_report(config, results_df, best_params, exp_dir):
    """Create summary report"""
    with open(os.path.join(exp_dir, 'summary_report.txt'), 'w') as f:
        f.write("Grid Search Cross-Validation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment: {config.name}\n")
        f.write(f"Description: {config.description}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of folds: {config.grid_search.n_folds}\n")
        f.write(f"Epochs per fold: {config.training.epochs}\n")
        f.write(f"Total parameter combinations: {len(results_df)}\n\n")
        
        f.write("Hyperparameter Grid:\n")
        f.write(f"Batch factors: {config.grid_search.batch_factors}\n")
        f.write(f"Loss functions: {config.grid_search.loss_functions}\n")
        f.write(f"Learning rates: {config.grid_search.learning_rates}\n\n")
        
        f.write("Best Parameters:\n")
        f.write(f"Batch size: {best_params['batch_size']} (factor: {best_params['batch_factor']})\n")
        f.write(f"Loss function: {best_params['loss_function']}\n")
        f.write(f"Learning rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"Average VAL MAE: {best_params['avg_val_mae']:.4f} ± {best_params['std_val_mae']:.4f}\n\n")
        
        f.write("Best Configuration Files:\n")
        f.write(f"  - best_config.json: Ready-to-use configuration with optimal parameters\n")
        f.write(f"  - best_config.yaml: Same configuration in YAML format\n\n")
        
        f.write("To reuse the best configuration:\n")
        f.write(f"  python train_with_config.py --config {os.path.join(exp_dir, 'best_config.json')}\n")


def main():
    parser = argparse.ArgumentParser(description='Grid search CV with configuration')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, choices=['quick_test', 'grid_search'],
                       help='Use a preset configuration')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.preset == 'quick_test':
        from config import get_quick_test_config
        config = get_quick_test_config()
        config.name = "cv_quick_test"
    else:
        from config import get_grid_search_config
        config = get_grid_search_config()
        config.name = "cv_grid_search"
    
    # Perform grid search
    perform_grid_search_cv(config)


if __name__ == "__main__":
    main()