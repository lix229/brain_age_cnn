"""
Configuration module for brain age CNN training
"""
import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class DataConfig:
    """Data configuration"""
    image_list_csv: str = "data/dataset_splits.csv"  # CSV with image names and ages
    image_base_dir: str = "./img"  # Base directory for images
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    image_size: tuple = (224, 224)
    normalize: bool = True
    augmentation: bool = False
    
    # CSV column names
    image_col: str = "image_name"
    age_col: str = "age"
    patient_id_col: str = "patient_id"
    split_col: Optional[str] = "split"  # Optional: predefined train/val/test splits


@dataclass
class ModelConfig:
    """Model configuration"""
    base_model_path: str = "model/DBN_VGG16.h5"
    finetune_layers: List[str] = field(default_factory=lambda: ["dense_3", "dropout_2", "dense_4"])
    finetune_vgg_blocks: int = 1  # Number of VGG blocks to unfreeze (from the end)
    freeze_backbone: bool = False  # If True, only finetune dense layers
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.0001
    optimizer: str = "adam"
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae"])
    
    # Learning rate schedule
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_mae"
    
    # Checkpointing
    checkpoint_monitor: str = "val_mae"
    checkpoint_mode: str = "min"
    save_best_only: bool = True


@dataclass
class GridSearchConfig:
    """Grid search configuration"""
    n_folds: int = 3
    batch_factors: List[float] = field(default_factory=lambda: np.linspace(1, 3, 1).tolist())
    loss_functions: List[str] = field(default_factory=lambda: ["mean_squared_error"])
    learning_rates: List[float] = field(default_factory=lambda: (7 * np.logspace(-6, -4, 3)).tolist())
    
    # Grid search specific settings
    shuffle_folds: bool = True
    stratify: bool = False  # For regression, might want to bin ages for stratification


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = "brain_age_experiment"
    description: str = ""
    output_dir: str = "experiments"
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    
    # Experiment tracking
    log_wandb: bool = False
    wandb_project: str = "brain-age-cnn"
    save_predictions: bool = True
    save_model_weights: bool = True
    
    def save(self, path: str):
        """
        Save configuration to file in JSON or YAML format.
        
        Args:
            path (str): File path to save configuration. Extension determines format:
                       .json for JSON format, .yaml/.yml for YAML format.
        
        Raises:
            ValueError: If file extension is not supported.
        """
        ext = os.path.splitext(path)[1].lower()
        config_dict = asdict(self)
        
        if ext == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load configuration from JSON or YAML file.
        
        Args:
            path (str): File path to load configuration from. Extension determines format:
                       .json for JSON format, .yaml/.yml for YAML format.
        
        Returns:
            ExperimentConfig: Loaded configuration object.
        
        Raises:
            ValueError: If file extension is not supported.
            FileNotFoundError: If configuration file doesn't exist.
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Handle nested dataclasses
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create ExperimentConfig from dictionary with nested dataclass handling.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
        
        Returns:
            ExperimentConfig: Configuration object with properly initialized nested dataclasses.
        """
        # Extract sub-configs
        data_config = DataConfig(**config_dict.pop('data', {}))
        model_config = ModelConfig(**config_dict.pop('model', {}))
        training_config = TrainingConfig(**config_dict.pop('training', {}))
        grid_search_config = GridSearchConfig(**config_dict.pop('grid_search', {}))
        
        # Create main config
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            grid_search=grid_search_config,
            **config_dict
        )
    
    def get_experiment_dir(self):
        """
        Generate unique experiment output directory with timestamp.
        
        Returns:
            str: Path to experiment directory in format: {output_dir}/{name}_{timestamp}
        """
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.output_dir, f"{self.name}_{timestamp}")
        return exp_dir
    
    def validate(self):
        """
        Validate configuration parameters and check if required files exist.
        
        Returns:
            bool: True if validation passes.
        
        Raises:
            FileNotFoundError: If base model file doesn't exist.
            ValueError: If split ratios are invalid or number of folds is too small.
        """
        # Check paths exist
        if not os.path.exists(self.data.image_list_csv):
            print(f"Warning: Image list CSV not found: {self.data.image_list_csv}")
            print("Make sure to create this file before training.")
        
        if not os.path.exists(self.data.image_base_dir):
            print(f"Warning: Image base directory not found: {self.data.image_base_dir}")
            print("This path should exist on the HiPerGator cluster.")
        
        if not os.path.exists(self.model.base_model_path):
            raise FileNotFoundError(f"Base model not found: {self.model.base_model_path}")
        
        # Validate splits
        total_split = self.data.validation_split + self.data.test_split
        if total_split >= 1.0:
            raise ValueError(f"Validation + test split must be < 1.0, got {total_split}")
        
        # Validate grid search
        if self.grid_search.n_folds < 2:
            raise ValueError("Number of folds must be >= 2")
        
        return True


# Preset configurations
def get_quick_test_config():
    """
    Get a preset configuration optimized for quick testing and development.
    
    Returns:
        ExperimentConfig: Configuration with minimal epochs (5), small batch size (4),
                         and reduced cross-validation folds (2) for fast iteration.
    """
    config = ExperimentConfig(
        name="quick_test",
        description="Quick test configuration with minimal epochs"
    )
    config.training.epochs = 5
    config.training.batch_size = 4
    config.grid_search.n_folds = 2
    return config


def get_full_training_config():
    """
    Get a preset configuration for full-scale training with production settings.
    
    Returns:
        ExperimentConfig: Configuration with 100 epochs, larger batch size (16),
                         and data augmentation enabled for optimal model performance.
    """
    config = ExperimentConfig(
        name="full_training",
        description="Full training configuration"
    )
    config.training.epochs = 100
    config.training.batch_size = 16
    config.data.augmentation = True
    return config


def get_grid_search_config():
    """
    Get a preset configuration for comprehensive hyperparameter grid search.
    
    Returns:
        ExperimentConfig: Configuration with expanded hyperparameter ranges:
                         - Batch factors: [0.5, 1.0, 2.0]
                         - Learning rates: 5 values from 1e-6 to 1e-3
                         - 3-fold cross-validation for robust evaluation
    """
    config = ExperimentConfig(
        name="grid_search",
        description="Comprehensive grid search configuration"
    )
    config.grid_search.batch_factors = [0.5, 1.0, 2.0]
    config.grid_search.learning_rates = (7 * np.logspace(-6, -3, 5)).tolist()
    return config