# Brain Age CNN - Deep Learning for Brain Age Prediction

## Overview

This project implements a deep learning pipeline for predicting brain age from medical imaging data using a fine-tuned DBN-VGG16 convolutional neural network. The system is designed for robust experimentation with comprehensive configuration management, cross-validation, and hyperparameter optimization.

## 🎯 Key Features

- **Configuration-Driven Training**: All experiments managed through JSON/YAML configuration files
- **Flexible Data Loading**: CSV-based image loading with support for predefined train/val/test splits
- **Model Finetuning**: Intelligent layer-wise finetuning of pre-trained DBN-VGG16 architecture
- **Cross-Validation**: K-fold cross-validation with comprehensive hyperparameter grid search
- **Production Ready**: Robust error handling, logging, and experiment tracking
- **HiPerGator Compatible**: Optimized for University of Florida's HiPerGator cluster

## 🏗️ System Architecture

### Core Components

```
Brain Age CNN Pipeline
├── Configuration System (config.py)
│   ├── DataConfig: Image paths, preprocessing, splits
│   ├── ModelConfig: Architecture, finetuning settings
│   ├── TrainingConfig: Optimization, callbacks, metrics
│   └── GridSearchConfig: Cross-validation, hyperparameters
│
├── Data Pipeline (data_loader_csv.py)
│   ├── CSV-based image loading
│   ├── Automatic train/val/test splitting
│   ├── Image preprocessing and augmentation
│   └── Both PyTorch and Keras compatibility
│
├── Training Pipeline
│   ├── Standard Training (train.py)
│   ├── Cross-Validation Grid Search
│   └── Model Architecture (DBN-VGG16 finetuning)
│
└── Inference & Analysis
    ├── Model Inference (inference.py)
    └── Results Analysis (analyze_cv_results.py)
```

### Model Architecture

The system employs a sophisticated finetuning strategy:

1. **Base Model**: Pre-trained DBN-VGG16 (Deep Belief Network + VGG16)
2. **Finetuning Strategy**: 
   - Last 3 dense layers: `dense_3`, `dropout_2`, `dense_4`
   - Optional VGG16 backbone unfreezing (configurable number of blocks)
3. **Output**: Single regression value (predicted age in years)
4. **Loss Function**: Mean Squared Error (MSE)
5. **Evaluation Metric**: Mean Absolute Error (MAE)

## 📁 Project Structure

```
brain_age_cnn/
├── 📁 Core Python Files
│   ├── train.py                       # Unified training script (main entry point)
│   ├── config.py                      # Configuration management system
│   ├── data_loader_generator.py       # Memory-efficient data loading
│   ├── inference.py                   # Model inference and evaluation
│   └── analyze_cv_results.py          # Results analysis and visualization
│
├── 📁 Data & Models
│   ├── data/
│   │   ├── dataset_splits.csv         # Main dataset CSV (user-provided)
│   │   └── dataset_splits_example.csv # Example CSV format
│   ├── model/
│   │   └── DBN_VGG16.h5              # Pre-trained base model
│
├── 📁 Output Directories (auto-created)
│   ├── experiments/                   # Training outputs
│   └── cv_results/                    # Cross-validation results
│
└── 📁 Documentation
    ├── README.md                      # This file
    └── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository and navigate to project directory
cd brain_age_cnn

# Install dependencies
pip install -r requirements.txt

# Verify required files exist
ls model/DBN_VGG16.h5  # Pre-trained model
ls data/dataset_splits.csv  # Your dataset CSV
```

### 2. Prepare Your Data

Create `data/dataset_splits.csv` with your brain imaging data:

```csv
patient_id,image_name,age,split
0001,slice_001.jpg,45.2,train
0002,slice_002.jpg,52.7,validation
0003,slice_003.jpg,38.9,test
```

**Required Columns:**
- `image_name`: Filename (relative to image directory)
- `age`: Ground truth age in years
- `patient_id`: Patient identifier
- `split`: train/validation/test (optional - auto-generated if missing)

### 3. Run Training

#### Unified Training Script (`train.py`)

The new unified training script combines all workflows:

```bash
# Quick test without grid search
python train.py --preset quick_test

# Full training with automatic hyperparameter optimization
python train.py --preset full_training --grid_search

# Memory-efficient training for large datasets (>100K images)
python train.py --preset memory_efficient

# Memory-efficient with grid search
python train.py --preset memory_efficient --grid_search

# Custom configuration
python train.py --config configs/custom_config.json --grid_search
```

#### Grid Search Integration

When `--grid_search` is enabled:
1. **Phase 1**: Performs k-fold cross-validation to find optimal hyperparameters
2. **Phase 2**: Automatically trains final model with best parameters
3. **Output**: Saves best configuration for future reuse

```bash
# Example with custom grid search parameters
python train.py --preset full_training --grid_search \
                --grid_batch_factors "0.5,1.0,2.0" \
                --grid_learning_rates "1e-5,5e-5,1e-4" \
                --grid_folds 5
```

### 4. Model Inference

```bash
# Single image prediction
python inference.py --model_path experiments/my_experiment_20250624_120000/best_model.h5 \
                   --image_path /path/to/image.jpg

# Batch processing
python inference.py --model_path experiments/my_experiment_20250624_120000/best_model.h5 \
                   --batch_dir /path/to/images/ \
                   --output_csv predictions.csv
```

## ⚙️ Configuration System

### Configuration Hierarchy

The configuration system uses nested dataclasses for organized parameter management:

```python
ExperimentConfig
├── data: DataConfig
│   ├── image_list_csv: "data/dataset_splits.csv"
│   ├── image_base_dir: "/blue/cruzalmeida/pvaldeshernandez/slices_for_deepbrainnet_new"
│   ├── validation_split: 0.2
│   ├── image_size: (224, 224)
│   └── augmentation: false
├── model: ModelConfig
│   ├── base_model_path: "model/DBN_VGG16.h5"
│   ├── finetune_layers: ["dense_3", "dropout_2", "dense_4"]
│   └── finetune_vgg_blocks: 1
├── training: TrainingConfig
│   ├── batch_size: 16
│   ├── epochs: 50
│   ├── learning_rate: 0.0001
│   └── early_stopping_patience: 10
└── grid_search: GridSearchConfig
    ├── n_folds: 3
    ├── batch_factors: [0.5, 1.0, 2.0]
    └── learning_rates: [1e-6, 1e-5, 1e-4, 1e-3]
```

### Creating Custom Configurations

```python
from config import ExperimentConfig

# Create custom configuration
config = ExperimentConfig(
    name="my_experiment",
    description="Custom brain age prediction experiment"
)

# Modify specific parameters
config.training.batch_size = 32
config.training.epochs = 75
config.training.learning_rate = 0.00005
config.model.finetune_vgg_blocks = 2

# Save configuration
config.save("configs/my_experiment.json")
```

### Preset Configurations

Three preset configurations are available:

1. **Quick Test** (`quick_test`): Fast iteration (5 epochs, small batch)
2. **Full Training** (`full_training`): Production settings (100 epochs, augmentation)
3. **Grid Search** (`grid_search`): Comprehensive hyperparameter search

## 🔬 Hyperparameter Optimization

### Integrated Grid Search

Grid search is now seamlessly integrated into the main training pipeline:

```bash
# Enable grid search with any preset
python train.py --preset full_training --grid_search
```

### Grid Search Parameters

- **Batch Size Factors**: `[0.5, 1.0, 2.0]` (multiplied by base batch size)
- **Learning Rates**: `[7e-6, 7e-5, 7e-4]` (logarithmic scale)
- **Loss Functions**: `["mean_squared_error"]` (extensible)
- **Cross-Validation**: 3-fold by default

### Workflow

1. **Grid Search Phase**:
   - Tests all parameter combinations
   - Uses k-fold cross-validation
   - Tracks performance metrics
   
2. **Training Phase**:
   - Automatically uses best parameters
   - Trains on full training set
   - Saves optimized configuration

### Results Analysis

```bash
# Analyze cross-validation results
python analyze_cv_results.py --results_dir cv_results/cv_gridsearch_20250624_120000

# Generates:
# - Learning rate vs MAE plots
# - Training time comparisons
# - Fold MAE distributions
# - Summary tables and reports
```

## 📊 Output Structure

### Standard Training Output
```
experiments/experiment_name_YYYYMMDD_HHMMSS/
├── config.json                    # Experiment configuration
├── best_model.h5                  # Best model weights
├── final_model.h5                 # Final model weights
├── training_history.csv           # Training metrics history
├── summary_report.txt             # Experiment summary
├── validation_predictions.csv     # Validation set predictions
└── test_predictions.csv          # Test set predictions (if available)
```

### Cross-Validation Output
```
cv_results/cv_gridsearch_YYYYMMDD_HHMMSS/
├── config.json                    # Grid search configuration
├── grid_search_results.csv        # All parameter combinations results
├── best_parameters.json           # Optimal hyperparameters
├── best_config.json              # Ready-to-use config with best parameters
├── best_config.yaml              # Same config in YAML format
├── summary_report.txt             # Grid search summary
├── analysis/                      # Generated visualizations
│   ├── lr_vs_mae.png
│   ├── training_times.png
│   └── results_summary.csv
└── params_X/                      # Individual parameter sets
    ├── parameters.json
    └── fold_Y/
        ├── best_model.h5
        └── history.csv
```

#### Reusing Best Configuration
After grid search, use the automatically generated best configuration:
```bash
# Train with optimal parameters found by grid search
python train_with_config.py --config cv_results/cv_gridsearch_20250624_120000/best_config.json
```

## 🧠 Model Details

### DBN-VGG16 Architecture

The model uses a sophisticated Deep Belief Network + VGG16 hybrid:

1. **VGG16 Backbone**: Pre-trained feature extraction
2. **Global Average Pooling**: Reduces spatial dimensions
3. **Dense Layers**: 
   - `dense_3`: 1024 units (ReLU activation)
   - `dropout_2`: 50% dropout for regularization
   - `dense_4`: 1 unit (linear output for age regression)

### Finetuning Strategy

- **Frozen Layers**: VGG16 convolutional layers (optionally unfrozen)
- **Trainable Layers**: Last 3 dense layers + optional VGG blocks
- **Learning Rate**: Reduced for stable finetuning (default: 1e-4)
- **Optimization**: Adam optimizer with learning rate scheduling

## 📈 Performance Monitoring

### Training Metrics
- **Loss**: Mean Squared Error (MSE)
- **Primary Metric**: Mean Absolute Error (MAE) in years
- **Validation**: Early stopping on validation MAE

### Callbacks
- **Model Checkpointing**: Save best model based on validation MAE
- **Early Stopping**: Prevent overfitting (patience: 10 epochs)
- **Learning Rate Reduction**: Adaptive LR scheduling
- **CSV Logging**: Detailed training history

## 🖥️ HiPerGator Integration

### Default Paths (HiPerGator)
- **Images**: `/blue/cruzalmeida/pvaldeshernandez/slices_for_deepbrainnet_new`
- **Dataset**: `data/dataset_splits.csv`
- **Model**: `model/DBN_VGG16.h5`

### Batch Job Example
```bash
#!/bin/bash
#SBATCH --job-name=brain_age_cnn
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1

module load tensorflow/2.8.0

# Full training with grid search (recommended)
python train.py --preset full_training --grid_search

# Memory-efficient for large datasets
python train.py --preset memory_efficient --grid_search
```

### Memory Considerations

For large datasets (>100K images), use memory-efficient mode:
- Loads images on-demand using generators
- Reduces memory usage from ~100GB to ~4-8GB
- Slightly slower per epoch but enables training on larger datasets

## 🔧 Advanced Usage

### Custom Data Pipeline
```python
from data_loader_csv import create_sample_csv, get_keras_datasets
from config import ExperimentConfig

# Create custom dataset CSV
create_sample_csv("data/my_dataset.csv", img_dir="/path/to/images")

# Load data with custom configuration
config = ExperimentConfig()
config.data.image_list_csv = "data/my_dataset.csv"
train_data, val_data, test_data = get_keras_datasets(config)
```

### Model Evaluation
```python
# Detailed evaluation with ground truth
python inference.py --model_path model.h5 \
                   --batch_dir images/ \
                   --output_csv predictions.csv \
                   --truth_csv ground_truth.csv

# Generates MAE, RMSE, and detailed error analysis
```

## 📚 Dependencies

### Core Requirements
- **TensorFlow** ≥ 2.6.0: Deep learning framework
- **NumPy** ≥ 1.19.0: Numerical computing
- **Pandas** ≥ 1.3.0: Data manipulation
- **Pillow** ≥ 8.0.0: Image processing
- **scikit-learn** ≥ 0.24.0: Machine learning utilities

### Visualization & Analysis
- **Matplotlib** ≥ 3.3.0: Plotting
- **Seaborn** ≥ 0.11.0: Statistical visualization
- **PyYAML** ≥ 5.4.0: YAML configuration support

## 🤝 Contributing

### Code Style
- Follow PEP 8 style guidelines
- Use comprehensive docstrings for all functions
- Include type hints where appropriate
- Add unit tests for new functionality

### Adding New Features
1. Create feature branch
2. Implement with proper documentation
3. Add configuration options if needed
4. Update relevant documentation
5. Submit pull request

## 📄 License

This project is developed for academic research purposes. Please cite appropriately if used in publications.

## 🆘 Support

### Common Issues
1. **File Not Found**: Ensure all paths in configuration are correct
2. **Memory Errors**: Reduce batch size in configuration
3. **GPU Issues**: Check CUDA compatibility and memory usage

### Getting Help
- Check configuration validation warnings
- Review training logs in experiment directories
- Examine error messages in detail
- Verify data format matches expected CSV structure

---

**Author**: Developed for brain age prediction research  
**Last Updated**: December 2024  
**Version**: 3.0 (Unified Training Pipeline)