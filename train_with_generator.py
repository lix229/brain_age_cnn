"""
Training script using memory-efficient data generators
"""
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import gc
import psutil

from config import ExperimentConfig
from data_loader_generator import get_generators, get_tf_datasets, estimate_memory_usage
from train_with_config import create_finetuned_model, create_summary_report


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def train_model_with_generator(config, use_tf_data=True):
    """
    Train model using memory-efficient data generators.
    
    Args:
        config: ExperimentConfig object
        use_tf_data: Whether to use tf.data.Dataset (True) or Keras Sequence (False)
    """
    # Validate configuration
    config.validate()
    
    # Create experiment directory
    exp_dir = config.get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(exp_dir, 'config.json'))
    
    print(f"\nInitial memory usage: {get_memory_usage():.2f} GB")
    
    # Load data generators
    print("\nCreating data generators...")
    if use_tf_data:
        train_data, val_data, test_data, train_df, val_df, test_df = get_tf_datasets(config)
        steps_per_epoch = len(train_df) // config.training.batch_size
        validation_steps = len(val_df) // config.training.batch_size
    else:
        train_gen, val_gen, test_gen, train_df, val_df, test_df = get_generators(config)
        train_data = train_gen
        val_data = val_gen
        test_data = test_gen
        steps_per_epoch = len(train_gen)
        validation_steps = len(val_gen)
    
    print(f"\nData statistics:")
    print(f"  Train: {len(train_df)} samples ({steps_per_epoch} batches)")
    print(f"  Validation: {len(val_df)} samples ({validation_steps} batches)")
    if test_df is not None:
        print(f"  Test: {len(test_df)} samples")
    
    print(f"\nMemory usage after data setup: {get_memory_usage():.2f} GB")
    
    # Get age statistics from dataframes
    print(f"\nAge statistics:")
    print(f"  Train: mean={train_df[config.data.age_col].mean():.1f}, "
          f"std={train_df[config.data.age_col].std():.1f}, "
          f"range=[{train_df[config.data.age_col].min():.1f}, {train_df[config.data.age_col].max():.1f}]")
    print(f"  Validation: mean={val_df[config.data.age_col].mean():.1f}, "
          f"std={val_df[config.data.age_col].std():.1f}, "
          f"range=[{val_df[config.data.age_col].min():.1f}, {val_df[config.data.age_col].max():.1f}]")
    
    # Create model
    print("\nCreating model...")
    model = create_finetuned_model(config)
    print(f"Memory usage after model creation: {get_memory_usage():.2f} GB")
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(exp_dir, 'best_model.h5')
    callbacks.append(keras.callbacks.ModelCheckpoint(
        checkpoint_path,
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
    
    # CSV logger
    csv_path = os.path.join(exp_dir, 'training_history.csv')
    callbacks.append(keras.callbacks.CSVLogger(csv_path))
    
    # Custom progress callback with memory monitoring
    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.start_time = None
            self.epoch_start_time = None
            self.best_val_mae = float('inf')
            self.best_epoch = 0
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print(f"\nTraining model for {self.epochs} epochs...")
            print(f"{'='*80}")
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            # Force garbage collection
            gc.collect()
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            total_elapsed = time.time() - self.start_time
            
            # Calculate ETA
            avg_epoch_time = total_elapsed / (epoch + 1)
            remaining_epochs = self.epochs - epoch - 1
            eta = avg_epoch_time * remaining_epochs
            
            # Progress bar
            progress = (epoch + 1) / self.epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Get metrics
            loss = logs.get('loss', 0)
            mae = logs.get('mae', 0)
            val_loss = logs.get('val_loss', 0)
            val_mae = logs.get('val_mae', 0)
            lr = logs.get('lr', 0)
            memory_gb = get_memory_usage()
            
            print(f"\rEpoch {epoch+1}/{self.epochs} [{bar}] {progress*100:.1f}% - "
                  f"Time: {epoch_time:.1f}s - ETA: {int(eta)}s - Memory: {memory_gb:.2f}GB")
            print(f"  loss: {loss:.4f} - mae: {mae:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - lr: {lr:.2e}")
            
            # Check for best epoch
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch + 1
                print(f"  ✓ New best validation MAE!")
                
        def on_train_end(self, logs=None):
            total_time = time.time() - self.start_time
            print(f"\n{'='*80}")
            print(f"Training completed in {int(total_time/60)}m {int(total_time%60)}s")
            print(f"Best validation MAE: {self.best_val_mae:.4f} at epoch {self.best_epoch}")
            print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    callbacks.append(ProgressCallback(config.training.epochs))
    
    # Train model
    try:
        history = model.fit(
            train_data,
            epochs=config.training.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch if use_tf_data else None,
            validation_steps=validation_steps if use_tf_data else None,
            verbose=0
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        print(f"Current memory usage: {get_memory_usage():.2f} GB")
        raise
    
    # Save final model
    if config.save_model_weights:
        final_model_path = os.path.join(exp_dir, 'final_model.h5')
        model.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")
    
    # Evaluate on test set if available
    if test_data is not None and test_df is not None:
        print("\nEvaluating on test set...")
        if use_tf_data:
            test_loss, test_mae = model.evaluate(test_data, verbose=0)
        else:
            test_loss, test_mae = model.evaluate(test_data, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.2f} years")
        
        # Save test results
        test_results = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae)
        }
        with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    # Save predictions if requested (using batches to avoid memory issues)
    if config.save_predictions:
        print("\nSaving predictions...")
        save_predictions_batch(model, val_df, config, exp_dir, 'validation')
        
        if test_df is not None:
            save_predictions_batch(model, test_df, config, exp_dir, 'test')
    
    # Create summary report
    create_summary_report(config, history, exp_dir)
    
    print(f"\nTraining complete! Results saved to: {exp_dir}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return model, history


def save_predictions_batch(model, df, config, exp_dir, dataset_name):
    """Save predictions in batches to avoid memory issues"""
    from data_loader_generator import BrainAgeDataGenerator
    
    print(f"  Generating {dataset_name} predictions...")
    
    # Create generator for predictions
    pred_gen = BrainAgeDataGenerator(df, config, shuffle=False, augment=False)
    
    # Predict in batches
    predictions = []
    for i in range(len(pred_gen)):
        X_batch, _ = pred_gen[i]
        pred_batch = model.predict(X_batch, verbose=0)
        predictions.extend(pred_batch.flatten())
    
    # Create dataframe
    pred_df = pd.DataFrame({
        'patient_id': df[config.data.patient_id_col].values if config.data.patient_id_col else range(len(df)),
        'true_age': df[config.data.age_col].values,
        'predicted_age': predictions,
        'error': np.array(predictions) - df[config.data.age_col].values
    })
    
    # Save
    output_path = os.path.join(exp_dir, f'{dataset_name}_predictions.csv')
    pred_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(pred_df)} {dataset_name} predictions")


def main():
    parser = argparse.ArgumentParser(description='Train brain age model with memory-efficient generators')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, choices=['quick_test', 'full_training'],
                       help='Use a preset configuration')
    parser.add_argument('--data_loader', type=str, choices=['tf_data', 'keras_sequence'],
                       default='tf_data', help='Type of data loader to use')
    
    # Override options
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--name', type=str, help='Override experiment name')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.preset:
        if args.preset == 'quick_test':
            from config import get_quick_test_config
            config = get_quick_test_config()
        else:
            from config import get_full_training_config
            config = get_full_training_config()
    else:
        config = ExperimentConfig()
    
    # Apply overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.name:
        config.name = args.name
    
    # Add generator tag to experiment name
    config.name = f"{config.name}_generator"
    
    # Train model
    use_tf_data = (args.data_loader == 'tf_data')
    train_model_with_generator(config, use_tf_data=use_tf_data)


if __name__ == "__main__":
    main()