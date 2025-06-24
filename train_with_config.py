"""
Training script using configuration module and CSV-based data loading
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

from config import ExperimentConfig
from data_loader_csv import get_keras_datasets


def create_finetuned_model(config):
    """
    Create and configure a finetuned model based on configuration settings.
    
    This function loads the pre-trained DBN_VGG16 model and configures it for 
    finetuning by unfreezing specified layers and setting up the optimizer.
    
    Args:
        config: ExperimentConfig object containing model and training configuration
        
    Returns:
        tf.keras.Model: Compiled model ready for training with:
            - Specified layers unfrozen for finetuning
            - Adam optimizer with configured learning rate
            - MSE loss and MAE metrics
            
    Raises:
        FileNotFoundError: If base model file doesn't exist
    """
    # Load base model
    base_model = keras.models.load_model(config.model.base_model_path, compile=False)
    
    print("\nModel architecture:")
    for i, layer in enumerate(base_model.layers):
        print(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")
    
    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze specified layers
    for layer_name in config.model.finetune_layers:
        for layer in base_model.layers:
            if layer.name == layer_name:
                layer.trainable = True
                print(f"Unfreezing layer: {layer_name}")
    
    # Unfreeze VGG blocks if specified
    if not config.model.freeze_backbone and config.model.finetune_vgg_blocks > 0:
        vgg16_model = base_model.layers[0]
        # VGG16 blocks are typically: block1, block2, block3, block4, block5
        # Each block has multiple conv layers
        n_layers_to_unfreeze = config.model.finetune_vgg_blocks * 4  # Approximate
        for layer in vgg16_model.layers[-n_layers_to_unfreeze:]:
            layer.trainable = True
            print(f"Unfreezing VGG layer: {layer.name}")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    
    base_model.compile(
        optimizer=optimizer,
        loss=config.training.loss_function,
        metrics=config.training.metrics
    )
    
    # Print summary of trainable parameters
    trainable_params = sum(tf.keras.backend.count_params(w) for w in base_model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in base_model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    
    return base_model


def train_model(config):
    """
    Train a brain age prediction model using the provided configuration.
    
    This function orchestrates the complete training pipeline including:
    - Data loading and preprocessing
    - Model creation and configuration
    - Training with callbacks (checkpointing, early stopping, etc.)
    - Evaluation and results saving
    
    Args:
        config: ExperimentConfig object containing all training parameters
        
    Returns:
        tuple: (model, history) where:
            - model: Trained Keras model
            - history: Training history object with loss/metric curves
            
    Raises:
        FileNotFoundError: If required files (model, data) don't exist
        ValueError: If configuration validation fails
    """
    # Validate configuration
    config.validate()
    
    # Create experiment directory
    exp_dir = config.get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(exp_dir, 'config.json'))
    
    # Load data
    print("\nLoading data...")
    (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, y_test, test_ids) = get_keras_datasets(config)
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, ages: {y_train.shape}")
    print(f"  Validation: {X_val.shape}, ages: {y_val.shape}")
    if X_test is not None:
        print(f"  Test: {X_test.shape}, ages: {y_test.shape}")
    
    print(f"\nAge statistics:")
    print(f"  Train: mean={y_train.mean():.1f}, std={y_train.std():.1f}, range=[{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  Validation: mean={y_val.mean():.1f}, std={y_val.std():.1f}, range=[{y_val.min():.1f}, {y_val.max():.1f}]")
    
    # Create model
    print("\nCreating model...")
    model = create_finetuned_model(config)
    
    # Create callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(exp_dir, 'best_model.h5')
    callbacks.append(keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=config.training.checkpoint_monitor,
        mode=config.training.checkpoint_mode,
        save_best_only=config.training.save_best_only,
        verbose=0  # Disable verbose to use custom progress
    ))
    
    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=config.training.early_stopping_monitor,
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=0  # Disable verbose to use custom progress
    ))
    
    # Reduce learning rate
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.training.reduce_lr_factor,
        patience=config.training.reduce_lr_patience,
        min_lr=config.training.min_lr,
        verbose=0  # Disable verbose to use custom progress
    ))
    
    # CSV logger
    csv_path = os.path.join(exp_dir, 'training_history.csv')
    callbacks.append(keras.callbacks.CSVLogger(csv_path))
    
    # Create custom progress callback
    class TrainingProgressCallback(keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.start_time = None
            self.epoch_start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            print(f"\nTraining model for {self.epochs} epochs...")
            print(f"{'='*80}")
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            
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
            
            # Format metrics
            loss = logs.get('loss', 0)
            mae = logs.get('mae', 0)
            val_loss = logs.get('val_loss', 0)
            val_mae = logs.get('val_mae', 0)
            lr = logs.get('lr', 0)
            
            print(f"\rEpoch {epoch+1}/{self.epochs} [{bar}] {progress*100:.1f}% - "
                  f"Time: {epoch_time:.1f}s - ETA: {int(eta)}s")
            print(f"  loss: {loss:.4f} - mae: {mae:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - lr: {lr:.2e}")
            
            # Check if this is the best epoch so far
            if hasattr(self, 'best_val_mae'):
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self.best_epoch = epoch + 1
                    print(f"  ✓ New best validation MAE!")
            else:
                self.best_val_mae = val_mae
                self.best_epoch = epoch + 1
                
        def on_train_end(self, logs=None):
            total_time = time.time() - self.start_time
            print(f"\n{'='*80}")
            print(f"Training completed in {int(total_time/60)}m {int(total_time%60)}s")
            print(f"Best validation MAE: {self.best_val_mae:.4f} at epoch {self.best_epoch}")
    
    # Add progress callback
    progress_callback = TrainingProgressCallback(config.training.epochs)
    callbacks.append(progress_callback)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0  # Disable default output since we have custom progress
    )
    
    # Save final model
    if config.save_model_weights:
        final_model_path = os.path.join(exp_dir, 'final_model.h5')
        model.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")
    
    # Evaluate on test set if available
    if X_test is not None:
        print("\nEvaluating on test set...")
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.2f} years")
        
        # Save test results
        test_results = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae)
        }
        with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    # Save predictions if requested
    if config.save_predictions:
        print("\nSaving predictions...")
        
        # Validation predictions with progress
        print("  Generating validation predictions...")
        val_preds = model.predict(X_val, verbose=0)
        val_df = pd.DataFrame({
            'patient_id': val_ids,
            'true_age': y_val,
            'predicted_age': val_preds.flatten(),
            'error': val_preds.flatten() - y_val
        })
        val_df.to_csv(os.path.join(exp_dir, 'validation_predictions.csv'), index=False)
        print(f"  ✓ Saved {len(val_df)} validation predictions")
        
        # Test predictions if available
        if X_test is not None:
            print("  Generating test predictions...")
            test_preds = model.predict(X_test, verbose=0)
            test_df = pd.DataFrame({
                'patient_id': test_ids,
                'true_age': y_test,
                'predicted_age': test_preds.flatten(),
                'error': test_preds.flatten() - y_test
            })
            test_df.to_csv(os.path.join(exp_dir, 'test_predictions.csv'), index=False)
            print(f"  ✓ Saved {len(test_df)} test predictions")
    
    # Create summary report
    create_summary_report(config, history, exp_dir)
    
    print(f"\nTraining complete! Results saved to: {exp_dir}")
    
    return model, history


def create_summary_report(config, history, exp_dir):
    """Create a summary report of the training"""
    report_path = os.path.join(exp_dir, 'summary_report.txt')
    
    # Get best epoch
    best_epoch = np.argmin(history.history['val_mae'])
    best_val_mae = history.history['val_mae'][best_epoch]
    
    with open(report_path, 'w') as f:
        f.write("Training Summary Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Experiment: {config.name}\n")
        f.write(f"Description: {config.description}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Base model: {config.model.base_model_path}\n")
        f.write(f"  Finetuned layers: {config.model.finetune_layers}\n")
        f.write(f"  Batch size: {config.training.batch_size}\n")
        f.write(f"  Learning rate: {config.training.learning_rate}\n")
        f.write(f"  Epochs: {config.training.epochs}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Best epoch: {best_epoch + 1}\n")
        f.write(f"  Best validation MAE: {best_val_mae:.4f} years\n")
        f.write(f"  Final training MAE: {history.history['mae'][-1]:.4f} years\n")
        f.write(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f} years\n")


def main():
    parser = argparse.ArgumentParser(description='Train brain age model with configuration')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, choices=['quick_test', 'full_training', 'grid_search'],
                       help='Use a preset configuration')
    
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
        elif args.preset == 'full_training':
            from config import get_full_training_config
            config = get_full_training_config()
        else:
            from config import get_grid_search_config
            config = get_grid_search_config()
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
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()