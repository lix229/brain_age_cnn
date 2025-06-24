"""
Data loader that reads image names and metadata from CSV file
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


class BrainAgeDatasetFromCSV(Dataset):
    """PyTorch Dataset that loads images based on CSV file"""
    
    def __init__(self, df, image_base_dir, image_col='image_name', 
                 age_col='age', patient_id_col='patient_id', transform=None):
        """
        Initialize PyTorch Dataset for brain age prediction from CSV data.
        
        Args:
            df (pd.DataFrame): DataFrame containing image metadata
            image_base_dir (str): Base directory path where images are stored
            image_col (str): Column name containing image filenames
            age_col (str): Column name containing age labels
            patient_id_col (str): Column name containing patient identifiers
            transform (torchvision.transforms): Optional transforms to apply to images
        """
        self.df = df.reset_index(drop=True)
        self.image_base_dir = image_base_dir
        self.image_col = image_col
        self.age_col = age_col
        self.patient_id_col = patient_id_col
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, age, patient_id) where:
                - image: Preprocessed image tensor
                - age: Age as float32 tensor
                - patient_id: Patient identifier string
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_base_dir, row[self.image_col])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get age and patient ID
        age = torch.tensor(row[self.age_col], dtype=torch.float32)
        patient_id = row[self.patient_id_col]
        
        return image, age, patient_id


def load_data_from_csv(config):
    """
    Load and split data from CSV file according to configuration settings.
    
    Args:
        config: ExperimentConfig object containing data configuration parameters
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for train, validation, and test sets
        
    Raises:
        ValueError: If required columns are missing from CSV
        FileNotFoundError: If CSV file doesn't exist
    """
    # Read CSV
    df = pd.read_csv(config.data.image_list_csv)
    
    # Validate columns exist
    required_cols = [config.data.image_col, config.data.age_col]
    if config.data.patient_id_col:
        required_cols.append(config.data.patient_id_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Check if predefined splits exist
    if config.data.split_col and config.data.split_col in df.columns:
        # Use predefined splits
        train_df = df[df[config.data.split_col] == 'train'].copy()
        val_df = df[df[config.data.split_col] == 'validation'].copy()
        test_df = df[df[config.data.split_col] == 'test'].copy()
        
        print(f"Using predefined splits from column '{config.data.split_col}':")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
    else:
        # Create splits
        # First split off test set
        if config.data.test_split > 0:
            train_val_df, test_df = train_test_split(
                df, test_size=config.data.test_split, 
                random_state=config.data.random_seed
            )
        else:
            train_val_df = df
            test_df = pd.DataFrame()
        
        # Then split train/val
        val_size = config.data.validation_split / (1 - config.data.test_split)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size,
            random_state=config.data.random_seed
        )
        
        print(f"Created random splits:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def get_keras_datasets(config):
    """
    Load and preprocess image datasets for Keras/TensorFlow training.
    
    Args:
        config: ExperimentConfig object containing data configuration
        
    Returns:
        tuple: Three tuples containing (X, y, ids) for train, validation, and test sets:
            - X: numpy array of preprocessed images (N, H, W, C)
            - y: numpy array of age labels (N,)
            - ids: list of patient identifiers
    """
    train_df, val_df, test_df = load_data_from_csv(config)
    
    def load_images_from_df(df):
        images = []
        ages = []
        patient_ids = []
        
        for _, row in df.iterrows():
            # Load image
            img_path = os.path.join(config.data.image_base_dir, row[config.data.image_col])
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
    
    # Load train data
    X_train, y_train, train_ids = load_images_from_df(train_df)
    X_val, y_val, val_ids = load_images_from_df(val_df)
    
    X_test, y_test, test_ids = None, None, None
    if len(test_df) > 0:
        X_test, y_test, test_ids = load_images_from_df(test_df)
    
    return (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, y_test, test_ids)


def get_pytorch_dataloaders(config):
    """Get PyTorch DataLoaders based on configuration"""
    train_df, val_df, test_df = load_data_from_csv(config)
    
    # Define transforms
    if config.data.normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Lambda(lambda x: x)
    
    # Training transforms (with augmentation if enabled)
    if config.data.augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = BrainAgeDatasetFromCSV(
        train_df, config.data.image_base_dir,
        config.data.image_col, config.data.age_col, 
        config.data.patient_id_col, train_transform
    )
    
    val_dataset = BrainAgeDatasetFromCSV(
        val_df, config.data.image_base_dir,
        config.data.image_col, config.data.age_col,
        config.data.patient_id_col, val_transform
    )
    
    test_dataset = None
    if len(test_df) > 0:
        test_dataset = BrainAgeDatasetFromCSV(
            test_df, config.data.image_base_dir,
            config.data.image_col, config.data.age_col,
            config.data.patient_id_col, val_transform
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def create_sample_csv(output_path, img_dir='img', n_samples=None):
    """Create a sample CSV file from images in a directory"""
    # Get all jpg files
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    if n_samples:
        image_files = image_files[:n_samples]
    
    # Extract patient IDs
    patient_ids = []
    for filename in image_files:
        import re
        match = re.search(r'sub-(\d+)', filename)
        if match:
            patient_ids.append(match.group(1))
        else:
            patient_ids.append('unknown')
    
    # Generate random ages for demo
    np.random.seed(42)
    ages = np.random.uniform(20, 80, len(image_files))
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'image_name': image_files,
        'age': ages,
        'split': np.random.choice(['train', 'validation', 'test'], 
                                 size=len(image_files), 
                                 p=[0.7, 0.2, 0.1])
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created sample CSV with {len(df)} images at: {output_path}")
    
    return df