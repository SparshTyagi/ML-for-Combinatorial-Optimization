import os
import torch

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
FIGURE_DIR = os.path.join(ROOT_DIR, 'figures')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODEL_DIR, FIGURE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Reduced graph generation parameters for faster execution
GRAPH_PARAMS = {
    'random': {
        'sizes': [20, 50, 100],  # Removed larger sizes
        'probabilities': [0.1, 0.5]  # Reduced probabilities
    },
    'scale_free': {
        'sizes': [20, 50, 100],  # Removed larger sizes
        'm': [2, 4]  # Reduced parameter values
    },
    'small_world': {
        'sizes': [20, 50, 100],  # Removed larger sizes
        'k': [4, 6],  # Reduced parameter values
        'p': [0.1, 0.5]  # Reduced parameter values
    },
    'geometric': {
        'sizes': [20, 50, 100],  # Removed larger sizes
        'radius': [0.2, 0.4]  # Reduced radius options
    }
}

# Dataset parameters
DATASET_PARAMS = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'stratify_by': 'num_nodes',
    'seed': 42
}

# Reduced model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 50,  # Reduced from 100
        'max_depth': 15,     # Reduced from 20
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'gnn': {
        'node_features': 4,
        'edge_features': 4,
        'hidden_channels': 32,  # Reduced from 64
        'num_layers': 2,        # Reduced from 3
        'gnn_type': 'gcn',
        'dropout': 0.2,
        'max_colors': 10
    },
    'hybrid': {
        'base_model_type': 'gnn',
        'ml_guidance_weight': 0.7,
        'local_search_iterations': 50  # Reduced from 100
    }
}

# Reduced training parameters
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 30,            # Reduced from 100
    'lr': 0.001,
    'weight_decay': 5e-4,
    'patience': 10,          # Reduced from 20
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Evaluation parameters
EVALUATION_PARAMS = {
    'batch_size': 64,
    'metrics': ['accuracy', 'computation_time', 'color_count_ratio']
}

# Baseline methods
BASELINE_METHODS = [
    'random_ordering',
    'degree_ordering'
    # Removed more complex methods to save time
]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(ROOT_DIR, 'edge_coloring_ml.log')
}