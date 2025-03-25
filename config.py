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

# Graph generation parameters
GRAPH_PARAMS = {
    'random': {
        'sizes': [20, 50, 100, 200, 500],
        'probabilities': [0.1, 0.3, 0.5, 0.8]
    },
    'scale_free': {
        'sizes': [20, 50, 100, 200, 500],
        'm': [2, 3, 4, 5]  # Number of edges to attach from a new node
    },
    'small_world': {
        'sizes': [20, 50, 100, 200, 500],
        'k': [4, 6, 8],  # Each node is connected to k nearest neighbors
        'p': [0.1, 0.3, 0.5]  # Rewiring probability
    },
    'geometric': {
        'sizes': [20, 50, 100, 200, 500],
        'radius': [0.2, 0.3, 0.4]  # Connection radius
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

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'gnn': {
        'node_features': 4,
        'edge_features': 4,
        'hidden_channels': 64,
        'num_layers': 3,
        'gnn_type': 'gcn',
        'dropout': 0.2,
        'max_colors': 10
    },
    'hybrid': {
        'base_model_type': 'gnn',
        'ml_guidance_weight': 0.7,
        'local_search_iterations': 100
    }
}

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'patience': 20,
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
    'degree_ordering',
    'vizing_implementation',
    'tabu_search'
]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(ROOT_DIR, 'edge_coloring_ml.log')
}