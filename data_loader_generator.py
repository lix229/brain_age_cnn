"""
Memory-efficient data loading using generators for large datasets
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import warnings


class BrainAgeDataGenerator(keras.utils.Sequence):
    """
    Memory-efficient data generator for brain age prediction.
    
    Loads images on-the-fly instead of loading all into memory at once.
    """
    
    def __init__(self, df, config, shuffle=True, augment=False):
        """
        Initialize the data generator.
        
        Args:
            df: DataFrame containing image paths and ages
            config: Configuration object
            shuffle: Whether to shuffle data after each epoch
            augment: Whether to apply data augmentation
        """
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = config.training.batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_size = tuple(config.data.image_size)
        self.normalize = config.data.normalize
        
        # Augmentation setup
        if augment and config.data.augmentation:
            self.augmentation = keras.Sequential([
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
                keras.layers.RandomTranslation(0.05, 0.05),
            ])
        else:
            self.augmentation = None
            
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices for the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        batch_df = self.df.iloc[indices]
        
        # Generate data
        X, y = self._generate_batch(batch_df)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, batch_df):
        """Generate batch of images and labels"""
        batch_size = len(batch_df)
        X = np.empty((batch_size, *self.image_size, 3), dtype=np.float32)
        y = np.empty(batch_size, dtype=np.float32)
        
        # Load images
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Load image
            img_path = os.path.join(self.config.data.image_base_dir, 
                                   row[self.config.data.image_col])
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img, dtype=np.float32)
                
                if self.normalize:
                    img_array = img_array / 255.0
                
                X[i] = img_array
                y[i] = row[self.config.data.age_col]
                
            except Exception as e:
                warnings.warn(f"Error loading image {img_path}: {e}")
                # Use black image as placeholder
                X[i] = np.zeros((*self.image_size, 3), dtype=np.float32)
                y[i] = 0
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            X = self.augmentation(X, training=True)
        
        return X, y


def get_generators(config):
    """
    Create data generators for training, validation, and test sets.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (train_generator, val_generator, test_generator, train_df, val_df, test_df)
    """
    from data_loader_csv import load_data_from_csv
    
    # Load dataframes
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_from_csv(config)
    
    # Create generators
    train_generator = BrainAgeDataGenerator(
        train_df, 
        config, 
        shuffle=True, 
        augment=config.data.augmentation
    )
    
    val_generator = BrainAgeDataGenerator(
        val_df, 
        config, 
        shuffle=False, 
        augment=False
    )
    
    test_generator = None
    if test_df is not None and len(test_df) > 0:
        test_generator = BrainAgeDataGenerator(
            test_df, 
            config, 
            shuffle=False, 
            augment=False
        )
    
    return train_generator, val_generator, test_generator, train_df, val_df, test_df


def estimate_memory_usage(config, train_df, val_df, test_df):
    """
    Estimate memory usage for loading all images at once.
    
    Args:
        config: Configuration object
        train_df, val_df, test_df: DataFrames with image information
        
    Returns:
        dict: Memory usage estimates
    """
    # Calculate per-image memory
    height, width = config.data.image_size
    channels = 3
    bytes_per_pixel = 4  # float32
    bytes_per_image = height * width * channels * bytes_per_pixel
    
    # Calculate total images
    n_train = len(train_df) if train_df is not None else 0
    n_val = len(val_df) if val_df is not None else 0
    n_test = len(test_df) if test_df is not None else 0
    n_total = n_train + n_val + n_test
    
    # Calculate memory usage
    train_memory_gb = (n_train * bytes_per_image) / (1024**3)
    val_memory_gb = (n_val * bytes_per_image) / (1024**3)
    test_memory_gb = (n_test * bytes_per_image) / (1024**3)
    total_memory_gb = (n_total * bytes_per_image) / (1024**3)
    
    return {
        'per_image_mb': bytes_per_image / (1024**2),
        'train_gb': train_memory_gb,
        'val_gb': val_memory_gb,
        'test_gb': test_memory_gb,
        'total_gb': total_memory_gb,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'n_total': n_total
    }


def create_tf_dataset(df, config, training=False):
    """
    Create a tf.data.Dataset for efficient data loading.
    
    Args:
        df: DataFrame with image paths and ages
        config: Configuration object
        training: Whether this is for training (enables shuffling/augmentation)
        
    Returns:
        tf.data.Dataset
    """
    def load_image(path, age):
        # Read and decode image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, config.data.image_size)
        
        if config.data.normalize:
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.cast(image, tf.float32)
            
        return image, age
    
    # Create dataset from dataframe
    paths = df[config.data.image_col].apply(
        lambda x: os.path.join(config.data.image_base_dir, x)
    ).values
    ages = df[config.data.age_col].values.astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, ages))
    
    # Map loading function
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply transformations
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        
        if config.data.augmentation:
            augmentation = keras.Sequential([
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.05),
                keras.layers.RandomTranslation(0.05, 0.05),
            ])
            dataset = dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    
    # Batch and prefetch
    dataset = dataset.batch(config.training.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_tf_datasets(config):
    """
    Create tf.data datasets for training, validation, and test sets.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (train_ds, val_ds, test_ds, train_df, val_df, test_df)
    """
    from data_loader_csv import load_data_from_csv
    
    # Load dataframes
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_from_csv(config)
    
    # Print memory estimates
    memory_info = estimate_memory_usage(config, train_df, val_df, test_df)
    print(f"\nMemory usage estimates:")
    print(f"  Per image: {memory_info['per_image_mb']:.2f} MB")
    print(f"  Training set: {memory_info['train_gb']:.2f} GB ({memory_info['n_train']} images)")
    print(f"  Validation set: {memory_info['val_gb']:.2f} GB ({memory_info['n_val']} images)")
    print(f"  Test set: {memory_info['test_gb']:.2f} GB ({memory_info['n_test']} images)")
    print(f"  Total (if loaded at once): {memory_info['total_gb']:.2f} GB")
    print(f"\nUsing memory-efficient data loading with generators...")
    
    # Create datasets
    train_ds = create_tf_dataset(train_df, config, training=True)
    val_ds = create_tf_dataset(val_df, config, training=False)
    
    test_ds = None
    if test_df is not None and len(test_df) > 0:
        test_ds = create_tf_dataset(test_df, config, training=False)
    
    return train_ds, val_ds, test_ds, train_df, val_df, test_df


if __name__ == "__main__":
    # Test the generator
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    config.data.image_base_dir = "./img"
    config.training.batch_size = 32
    
    # Test generator creation
    try:
        train_gen, val_gen, test_gen, train_df, val_df, test_df = get_generators(config)
        print(f"\nGenerator test successful!")
        print(f"Train batches: {len(train_gen)}")
        print(f"Val batches: {len(val_gen)}")
        if test_gen:
            print(f"Test batches: {len(test_gen)}")
            
        # Test loading one batch
        print("\nTesting batch loading...")
        X_batch, y_batch = train_gen[0]
        print(f"Batch shape: {X_batch.shape}")
        print(f"Age range in batch: {y_batch.min():.1f} - {y_batch.max():.1f}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()