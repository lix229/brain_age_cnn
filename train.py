"""
Unified training script with integrated grid search and memory-efficient loading
"""
import os
import argparse
import json
import time
import gc
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

from config import (ExperimentConfig, get_quick_test_config, 
                   get_full_training_config, get_memory_efficient_config)
from data_loader_generator import get_generators, get_tf_datasets
from data_loader_csv import load_data_from_csv


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def create_finetuned_model(config, learning_rate=None):
    """Create model for finetuning based on configuration"""
    if learning_rate is None:
        learning_rate = config.training.learning_rate
        
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


def perform_grid_search(config):
    """Perform grid search with cross-validation"""
    print_banner("GRID SEARCH PHASE")
    
    # Load data
    print("\nLoading data for grid search...")
    train_df, val_df, test_df = load_data_from_csv(config)
    
    # Combine train and validation for CV
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Create data generator for memory efficiency
    from data_loader_generator import BrainAgeDataGenerator
    data_gen = BrainAgeDataGenerator(full_train_df, config, shuffle=False, augment=False)
    
    # Load all data for cross-validation
    print("Loading images for cross-validation...")
    images = []
    ages = []
    for i in range(len(data_gen)):
        X_batch, y_batch = data_gen[i]
        images.extend(X_batch)
        ages.extend(y_batch)
    images = np.array(images)
    ages = np.array(ages)
    
    print(f"Loaded {len(images)} images for cross-validation")
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f} years")
    
    # Define hyperparameter grid
    batch_factors = config.grid_search.batch_factors
    learning_rates = config.grid_search.learning_rates
    base_batch_size = config.training.batch_size
    
    total_combinations = len(batch_factors) * len(learning_rates)
    print(f"\nTotal parameter combinations: {total_combinations}")
    print(f"Cross-validation folds: {config.grid_search.n_folds}")
    
    # Initialize results storage
    results = []
    
    # Create KFold object
    kf = KFold(n_splits=config.grid_search.n_folds, 
               shuffle=config.grid_search.shuffle_folds, 
               random_state=config.data.random_seed)
    
    overall_start_time = time.time()
    param_id = 0
    
    # Grid search
    for batch_factor, lr in product(batch_factors, learning_rates):
        param_id += 1
        batch_size = int(base_batch_size * batch_factor)
        
        elapsed_time = time.time() - overall_start_time
        if param_id > 1:
            eta_seconds = (elapsed_time / (param_id - 1)) * (total_combinations - param_id + 1)
            eta_str = f"{int(eta_seconds/60)}m {int(eta_seconds%60)}s"
        else:
            eta_str = "Calculating..."
            
        print(f"\n{'='*70}")
        print(f"PARAMETER SET {param_id}/{total_combinations}")
        print(f"Progress: {(param_id-1)/total_combinations*100:.1f}% - ETA: {eta_str}")
        print(f"Batch size: {batch_size}, Learning rate: {lr:.6f}")
        print(f"{'='*70}")
        
        # Cross-validation
        fold_maes = []
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(images), 1):
            print(f"\n  Fold {fold_num}/{config.grid_search.n_folds}: Training...")
            
            # Split data
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = ages[train_idx], ages[val_idx]
            
            # Create and train model
            model = create_finetuned_model(config, lr)
            
            # Simple training without extensive callbacks for grid search
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=config.training.epochs,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Get best validation MAE
            best_val_mae = min(history.history['val_mae'])
            fold_maes.append(best_val_mae)
            print(f"  Fold {fold_num} - Best VAL MAE: {best_val_mae:.4f}")
            
            # Clean up
            del model
            gc.collect()
        
        # Calculate average performance
        avg_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        
        print(f"\nAverage VAL MAE: {avg_mae:.4f} ± {std_mae:.4f}")
        
        # Store results
        result = {
            'batch_size': batch_size,
            'batch_factor': batch_factor,
            'learning_rate': lr,
            'avg_val_mae': avg_mae,
            'std_val_mae': std_mae,
            'fold_maes': fold_maes
        }
        results.append(result)
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['avg_val_mae'].idxmin()
    best_params = results_df.iloc[best_idx]
    
    total_time = time.time() - overall_start_time
    print_banner("GRID SEARCH COMPLETE")
    print(f"Total time: {int(total_time/60)}m {int(total_time%60)}s")
    print(f"\nBest parameters found:")
    print(f"  - Batch size: {best_params['batch_size']}")
    print(f"  - Learning rate: {best_params['learning_rate']:.6f}")
    print(f"  - Validation MAE: {best_params['avg_val_mae']:.4f} ± {best_params['std_val_mae']:.4f}")
    
    return best_params


def train_model(config, use_generator=True):
    """Train model with given configuration"""
    
    # Create experiment directory
    exp_dir = config.get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(exp_dir, 'config.json'))
    
    print(f"\nMemory usage at start: {get_memory_usage():.2f} GB")
    
    # Load data
    if use_generator:
        print("\nUsing memory-efficient data generators...")
        train_data, val_data, test_data, train_df, val_df, test_df = get_tf_datasets(config)
        steps_per_epoch = len(train_df) // config.training.batch_size
        validation_steps = len(val_df) // config.training.batch_size
    else:
        print("\nLoading all data into memory...")
        from data_loader_csv import get_keras_datasets
        (X_train, y_train, _), (X_val, y_val, _), test_data = get_keras_datasets(config)
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        steps_per_epoch = None
        validation_steps = None
    
    # Create model
    print("\nCreating model...")
    model = create_finetuned_model(config)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint
    callbacks.append(keras.callbacks.ModelCheckpoint(
        os.path.join(exp_dir, 'best_model.h5'),
        monitor=config.training.checkpoint_monitor,
        mode=config.training.checkpoint_mode,
        save_best_only=config.training.save_best_only,
        verbose=0
    ))
    
    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=config.training.early_stopping_monitor,
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=0
    ))
    
    # Reduce learning rate
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.training.reduce_lr_factor,
        patience=config.training.reduce_lr_patience,
        min_lr=config.training.min_lr,
        verbose=0
    ))
    
    # Progress callback
    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print(f"\nTraining for {self.epochs} epochs...")
            
        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self.start_time
            eta = (elapsed / (epoch + 1)) * (self.epochs - epoch - 1)
            
            progress = (epoch + 1) / self.epochs * 100
            mae = logs.get('mae', 0)
            val_mae = logs.get('val_mae', 0)
            
            print(f"\rEpoch {epoch+1}/{self.epochs} - {progress:.1f}% - "
                  f"MAE: {mae:.4f} - Val MAE: {val_mae:.4f} - "
                  f"ETA: {int(eta)}s - Memory: {get_memory_usage():.2f}GB", 
                  end='', flush=True)
            
            if epoch == self.epochs - 1:
                print()  # New line at end
    
    callbacks.append(ProgressCallback(config.training.epochs))
    
    # Train model
    print_banner("TRAINING MODEL")
    
    if use_generator:
        history = model.fit(
            train_data,
            epochs=config.training.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=0
        )
    else:
        history = model.fit(
            train_data[0], train_data[1],
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0
        )
    
    # Save results
    if config.save_model_weights:
        model.save(os.path.join(exp_dir, 'final_model.h5'))
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(exp_dir, 'training_history.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(exp_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Configuration: {config.name}\n")
        f.write(f"Description: {config.description}\n")
        f.write(f"Best epoch: {np.argmin(history.history['val_mae']) + 1}\n")
        f.write(f"Best validation MAE: {min(history.history['val_mae']):.4f}\n")
        f.write(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}\n")
    
    print(f"\nResults saved to: {exp_dir}")
    
    return model, history


def run_training_pipeline(config, enable_grid_search=False, use_generator=True):
    """Run the complete training pipeline"""
    start_time = time.time()
    
    # Step 1: Optional Grid Search
    if enable_grid_search:
        best_params = perform_grid_search(config)
        
        # Update config with best parameters
        config.training.batch_size = int(best_params['batch_size'])
        config.training.learning_rate = float(best_params['learning_rate'])
        
        # Update experiment name
        original_name = config.name
        config.name = f"{original_name}_best"
        
        # Save best configuration
        os.makedirs("configs/best", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_config_path = f"configs/best/config_{timestamp}.json"
        config.save(best_config_path)
        print(f"\nBest configuration saved to: {best_config_path}")
    
    # Step 2: Train final model
    print_banner("FINAL MODEL TRAINING")
    model, history = train_model(config, use_generator)
    
    # Summary
    total_time = time.time() - start_time
    print_banner("TRAINING COMPLETE")
    print(f"Total pipeline time: {int(total_time/60)}m {int(total_time%60)}s")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Unified brain age prediction training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config', type=str, help='Path to configuration file')
    config_group.add_argument('--preset', type=str, 
                            choices=['quick_test', 'full_training', 'memory_efficient'],
                            default='memory_efficient',
                            help='Use a preset configuration')
    
    # Training options
    parser.add_argument('--grid_search', action='store_true',
                       help='Enable grid search for hyperparameter optimization')
    parser.add_argument('--no_generator', action='store_true',
                       help='Disable memory-efficient generators')
    
    # Override options
    parser.add_argument('--name', type=str, help='Override experiment name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    
    # Grid search options
    parser.add_argument('--grid_batch_factors', type=str,
                       help='Comma-separated batch factors')
    parser.add_argument('--grid_learning_rates', type=str,
                       help='Comma-separated learning rates')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        if args.preset == 'quick_test':
            config = get_quick_test_config()
        elif args.preset == 'full_training':
            config = get_full_training_config()
        else:
            config = get_memory_efficient_config()
    
    # Apply overrides
    if args.name:
        config.name = args.name
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Apply grid search overrides
    if args.grid_batch_factors:
        config.grid_search.batch_factors = [float(x) for x in args.grid_batch_factors.split(',')]
    if args.grid_learning_rates:
        config.grid_search.learning_rates = [float(x) for x in args.grid_learning_rates.split(',')]
    
    # Print configuration
    print_banner("BRAIN AGE PREDICTION TRAINING")
    print(f"Configuration: {args.preset if not args.config else args.config}")
    print(f"Grid Search: {'Enabled' if args.grid_search else 'Disabled'}")
    print(f"Memory-Efficient: {'Yes' if not args.no_generator else 'No'}")
    
    # Run training pipeline
    run_training_pipeline(
        config,
        enable_grid_search=args.grid_search,
        use_generator=not args.no_generator
    )


if __name__ == "__main__":
    main()