import os
import argparse
import time
import logging
import json
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src.utils import setup_logging, set_seed, timer, save_results
from src.graph_generation.random_graphs import generate_random_graph
from src.graph_generation.scale_free_graphs import generate_scale_free_graph
from src.graph_generation.small_world_graphs import generate_small_world_graph
from src.graph_generation.geometric_graphs import generate_geometric_graph
from src.coloring.greedy import greedy_edge_coloring
from src.coloring.vizing import vizing_edge_coloring
from src.coloring.ilp import ilp_edge_coloring
from src.coloring.local_search import local_search_coloring
from src.features.graph_features import extract_graph_features
from src.features.edge_features import extract_edge_features
from src.training.dataset import EdgeColoringDataset, TabularEdgeColoringDataset
from src.training.train_evaluate import train_model, evaluate_model
from src.models.random_forest import EdgeColoringRandomForest
from src.models.gnn import GNNEdgeColoring
from src.models.hybrid_model import HybridEdgeColoring
from src.visualization.graph_viz import visualize_coloring as visualize_graph_coloring
from src.visualization.results_viz import (
    plot_performance_comparison,
    plot_feature_importance,
    plot_scaling_behavior,
    plot_solution_quality
)
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Edge Coloring ML Framework')
    
    # Main operation modes
    parser.add_argument('--generate-graphs', action='store_true', help='Generate graph instances')
    parser.add_argument('--generate-colorings', action='store_true', help='Generate edge colorings')
    parser.add_argument('--extract-features', action='store_true', help='Extract graph and edge features')
    parser.add_argument('--prepare-datasets', action='store_true', help='Prepare datasets for training')
    parser.add_argument('--train-models', action='store_true', help='Train machine learning models')
    parser.add_argument('--evaluate-models', action='store_true', help='Evaluate trained models')
    parser.add_argument('--visualize-results', action='store_true', help='Visualize results')
    parser.add_argument('--run-all', action='store_true', help='Run the complete pipeline')
    
    # Graph generation parameters
    parser.add_argument('--graph-types', nargs='+', choices=['random', 'scale_free', 'small_world', 'geometric'], 
                        default=['random', 'scale_free', 'small_world', 'geometric'],
                        help='Types of graphs to generate')
    parser.add_argument('--num-per-config', type=int, default=5, 
                        help='Number of graphs to generate per configuration')
    
    # Coloring algorithms
    parser.add_argument('--coloring-methods', nargs='+', 
                        choices=['random_ordering', 'degree_ordering', 'vizing', 'ilp', 'local_search'],
                        default=['random_ordering', 'degree_ordering'], 
                        help='Methods to use for generating edge colorings')
    
    # Model selection
    parser.add_argument('--models', nargs='+', choices=['random_forest', 'gnn', 'hybrid'],
                        default=['random_forest', 'gnn', 'hybrid'], 
                        help='Machine learning models to train and evaluate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, 
                        help='Batch size for training (overrides config)')
    parser.add_argument('--lr', type=float, default=None, 
                        help='Learning rate (overrides config)')
    
    # Evaluation parameters
    parser.add_argument('--eval-graph-types', nargs='+', 
                        choices=['random', 'scale_free', 'small_world', 'geometric', 'real_world'],
                        default=['random', 'scale_free', 'small_world', 'geometric'], 
                        help='Types of graphs to use for evaluation')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--no-cuda', action='store_true', 
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


@timer
def generate_graph_instances(args):
    """Generate graph instances."""
    logger = logging.getLogger('graph_generation')
    logger.info('Generating graph instances...')
    
    graphs = {}
    for graph_type in args.graph_types:
        logger.info(f'Generating {graph_type} graphs...')
        type_params = config.GRAPH_PARAMS[graph_type]
        
        # Extract parameters for this graph type
        sizes = type_params['sizes']
        
        if graph_type == 'random':
            probabilities = type_params['probabilities']
            for size in sizes:
                for p in probabilities:
                    key = f'{graph_type}_n{size}_p{p}'
                    graphs[key] = []
                    for i in range(args.num_per_config):
                        # Use the imported generator
                        G = generate_random_graph(size, p, seed=args.seed + i, connected=True)
                        graphs[key].append(G)
                        
        elif graph_type == 'scale_free':
            m_values = type_params['m']
            for size in sizes:
                for m in m_values:
                    key = f'{graph_type}_n{size}_m{m}'
                    graphs[key] = []
                    for i in range(args.num_per_config):
                        # Use the imported generator
                        G = generate_scale_free_graph(size, m, seed=args.seed + i)
                        graphs[key].append(G)
                        
        elif graph_type == 'small_world':
            k_values = type_params['k']
            p_values = type_params['p']
            for size in sizes:
                for k in k_values:
                    for p in p_values:
                        key = f'{graph_type}_n{size}_k{k}_p{p}'
                        graphs[key] = []
                        for i in range(args.num_per_config):
                            # Use the imported generator
                            G = generate_small_world_graph(size, k, p, seed=args.seed + i)
                            graphs[key].append(G)
                            
        elif graph_type == 'geometric':
            radius_values = type_params['radius']
            for size in sizes:
                for radius in radius_values:
                    key = f'{graph_type}_n{size}_r{radius}'
                    graphs[key] = []
                    for i in range(args.num_per_config):
                        # Use the imported generator
                        G = generate_geometric_graph(size, radius, seed=args.seed + i, ensure_connected=True)
                        graphs[key].append(G)
    
    # Save the generated graphs
    for key, graph_list in graphs.items():
        save_dir = os.path.join(config.RAW_DATA_DIR, 'graphs', key)
        os.makedirs(save_dir, exist_ok=True)
        for i, G in enumerate(graph_list):
            nx.write_graphml(G, os.path.join(save_dir, f'graph_{i}.graphml'))
    
    logger.info(f'Generated {sum(len(g) for g in graphs.values())} graphs across {len(graphs)} configurations')
    return graphs


@timer
def generate_edge_colorings(args, graphs=None):
    """Generate edge colorings for the graphs."""
    logger = logging.getLogger('coloring_generation')
    logger.info('Generating edge colorings...')
    
    if graphs is None:
        # Load graphs from files
        graphs = {}
        graph_dirs = [d for d in os.listdir(os.path.join(config.RAW_DATA_DIR, 'graphs')) 
                     if os.path.isdir(os.path.join(config.RAW_DATA_DIR, 'graphs', d))]
        
        for graph_dir in graph_dirs:
            graphs[graph_dir] = []
            dir_path = os.path.join(config.RAW_DATA_DIR, 'graphs', graph_dir)
            for filename in sorted(os.listdir(dir_path)):
                if filename.endswith('.graphml'):
                    G = nx.read_graphml(os.path.join(dir_path, filename))
                    # Convert node labels back to integers if needed
                    if not isinstance(list(G.nodes())[0], int):
                        mapping = {n: int(n) for n in G.nodes()}
                        G = nx.relabel_nodes(G, mapping)
                    graphs[graph_dir].append(G)
    
    colorings = {}
    for graph_key, graph_list in graphs.items():
        colorings[graph_key] = {}
        logger.info(f'Generating colorings for {graph_key}...')
        
        for method in args.coloring_methods:
            colorings[graph_key][method] = []
            
            for i, G in enumerate(graph_list):
                if method == 'random_ordering':
                    coloring = greedy_edge_coloring(G, 'random')
                elif method == 'degree_ordering':
                    coloring = greedy_edge_coloring(G, 'degree')
                elif method == 'vizing':
                    coloring = vizing_edge_coloring(G)
                elif method == 'ilp':
                    # Only use ILP for small graphs
                    if G.number_of_nodes() <= 50:
                        coloring = ilp_edge_coloring(G)
                    else:
                        logger.warning(f'Skipping ILP for large graph {graph_key} #{i}')
                        continue
                elif method == 'local_search':
                    # Start with a greedy coloring and improve it
                    initial_coloring = greedy_edge_coloring(G, 'degree')
                    coloring = local_search_coloring(G, initial_coloring)
                
                colorings[graph_key][method].append(coloring)
                
                # Log some statistics
                if i == 0:  # Just log for the first graph in each group
                    num_colors = len(set(coloring.values()))
                    max_degree = max(dict(G.degree()).values())
                    logger.info(f'{method} used {num_colors} colors (max degree: {max_degree})')
    
    # Save the generated colorings
    for graph_key, method_colorings in colorings.items():
        for method, coloring_list in method_colorings.items():
            save_dir = os.path.join(config.RAW_DATA_DIR, 'colorings', graph_key, method)
            os.makedirs(save_dir, exist_ok=True)
            for i, coloring in enumerate(coloring_list):
                with open(os.path.join(save_dir, f'coloring_{i}.json'), 'w') as f:
                    # Convert keys to strings for JSON serialization
                    serializable_coloring = {str(k): v for k, v in coloring.items()}
                    json.dump(serializable_coloring, f)
    
    return colorings


@timer
def extract_features_for_graphs(args, graphs=None, colorings=None):
    """Extract features for graphs and edges."""
    logger = logging.getLogger('feature_extraction')
    logger.info('Extracting features...')
    
    if graphs is None or colorings is None:
        # Load graphs and colorings from files
        graphs = {}
        colorings = {}
        
        graph_dirs = [d for d in os.listdir(os.path.join(config.RAW_DATA_DIR, 'graphs')) 
                     if os.path.isdir(os.path.join(config.RAW_DATA_DIR, 'graphs', d))]
        
        for graph_dir in graph_dirs:
            graphs[graph_dir] = []
            dir_path = os.path.join(config.RAW_DATA_DIR, 'graphs', graph_dir)
            for filename in sorted(os.listdir(dir_path)):
                if filename.endswith('.graphml'):
                    G = nx.read_graphml(os.path.join(dir_path, filename))
                    # Convert node labels back to integers if needed
                    if not isinstance(list(G.nodes())[0], int):
                        mapping = {n: int(n) for n in G.nodes()}
                        G = nx.relabel_nodes(G, mapping)
                    graphs[graph_dir].append(G)
            
            colorings[graph_dir] = {}
            coloring_methods = os.listdir(os.path.join(config.RAW_DATA_DIR, 'colorings', graph_dir))
            for method in coloring_methods:
                colorings[graph_dir][method] = []
                method_path = os.path.join(config.RAW_DATA_DIR, 'colorings', graph_dir, method)
                for filename in sorted(os.listdir(method_path)):
                    if filename.endswith('.json'):
                        with open(os.path.join(method_path, filename), 'r') as f:
                            coloring_data = json.load(f)
                            # Convert string keys back to tuples
                            coloring = {}
                            for k, v in coloring_data.items():
                                # Keys are stored as string representations of tuples like '(0, 1)'
                                if k.startswith('(') and k.endswith(')'):
                                    k = eval(k)
                                coloring[k] = v
                            colorings[graph_dir][method].append(coloring)
    
    # Extract features
    graph_features = {}
    edge_features = {}
    
    for graph_key, graph_list in graphs.items():
        graph_features[graph_key] = []
        edge_features[graph_key] = []
        
        for i, G in enumerate(graph_list):
            # Extract graph-level features
            g_features = extract_graph_features(G)
            graph_features[graph_key].append(g_features)
            
            # Extract edge-level features
            e_features = extract_edge_features(G)
            edge_features[graph_key].append(e_features)
    
    # Save features
    os.makedirs(os.path.join(config.PROCESSED_DATA_DIR, 'features'), exist_ok=True)
    
    with open(os.path.join(config.PROCESSED_DATA_DIR, 'features', 'graph_features.json'), 'w') as f:
        # Convert to serializable format
        serializable_features = {}
        for k, v in graph_features.items():
            serializable_features[k] = [dict(item) for item in v]
        json.dump(serializable_features, f)
    
    with open(os.path.join(config.PROCESSED_DATA_DIR, 'features', 'edge_features.json'), 'w') as f:
        # Convert to serializable format
        serializable_features = {}
        for k, v in edge_features.items():
            serializable_features[k] = []
            for graph_edges in v:
                edge_dict = {}
                for edge, features in graph_edges.items():
                    edge_str = str(edge)
                    edge_dict[edge_str] = list(features)
                serializable_features[k].append(edge_dict)
        json.dump(serializable_features, f)
    
    logger.info(f'Extracted features for {sum(len(g) for g in graphs.values())} graphs')
    return graph_features, edge_features


@timer
def prepare_datasets(args, graphs=None, colorings=None, graph_features=None, edge_features=None):
    """Prepare datasets for training."""
    logger = logging.getLogger('dataset_preparation')
    logger.info('Preparing datasets...')
    
    if graphs is None or colorings is None or graph_features is None or edge_features is None:
        # Load data from files
        graphs = {}
        colorings = {}
        
        # Load graphs
        graph_dirs = [d for d in os.listdir(os.path.join(config.RAW_DATA_DIR, 'graphs')) 
                     if os.path.isdir(os.path.join(config.RAW_DATA_DIR, 'graphs', d))]
        
        for graph_dir in graph_dirs:
            graphs[graph_dir] = []
            dir_path = os.path.join(config.RAW_DATA_DIR, 'graphs', graph_dir)
            for filename in sorted(os.listdir(dir_path)):
                if filename.endswith('.graphml'):
                    G = nx.read_graphml(os.path.join(dir_path, filename))
                    # Convert node labels back to integers if needed
                    if not isinstance(list(G.nodes())[0], int):
                        mapping = {n: int(n) for n in G.nodes()}
                        G = nx.relabel_nodes(G, mapping)
                    graphs[graph_dir].append(G)
            
            # Load colorings
            colorings[graph_dir] = {}
            coloring_methods = os.listdir(os.path.join(config.RAW_DATA_DIR, 'colorings', graph_dir))
            for method in coloring_methods:
                colorings[graph_dir][method] = []
                method_path = os.path.join(config.RAW_DATA_DIR, 'colorings', graph_dir, method)
                for filename in sorted(os.listdir(method_path)):
                    if filename.endswith('.json'):
                        with open(os.path.join(method_path, filename), 'r') as f:
                            coloring_data = json.load(f)
                            # Convert string keys back to tuples
                            coloring = {}
                            for k, v in coloring_data.items():
                                # Keys are stored as string representations of tuples like '(0, 1)'
                                if k.startswith('(') and k.endswith(')'):
                                    k = eval(k)
                                coloring[k] = v
                            colorings[graph_dir][method].append(coloring)
        
        # Load features
        with open(os.path.join(config.PROCESSED_DATA_DIR, 'features', 'graph_features.json'), 'r') as f:
            graph_features_data = json.load(f)
            graph_features = {}
            for k, v in graph_features_data.items():
                graph_features[k] = [pd.Series(item) for item in v]
        
        with open(os.path.join(config.PROCESSED_DATA_DIR, 'features', 'edge_features.json'), 'r') as f:
            edge_features_data = json.load(f)
            edge_features = {}
            for k, v in edge_features_data.items():
                edge_features[k] = []
                for graph_edges in v:
                    edge_dict = {}
                    for edge_str, features in graph_edges.items():
                        edge = eval(edge_str) if edge_str.startswith('(') else edge_str
                        edge_dict[edge] = np.array(features)
                    edge_features[k].append(edge_dict)
    
    # Prepare PyTorch Geometric dataset for GNN models
    logger.info('Preparing PyTorch Geometric dataset...')
    all_graphs = []
    all_colorings = []
    
    for graph_key, graph_list in graphs.items():
        for method, coloring_list in colorings[graph_key].items():
            for i, (G, coloring) in enumerate(zip(graph_list, coloring_list)):
                # Skip if coloring is empty (e.g., ILP timed out for large graphs)
                if not coloring:
                    continue
                
                all_graphs.append(G)
                all_colorings.append(coloring)
    
    # Create and save dataset for PyTorch Geometric
    dataset_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'pytorch_geometric')
    os.makedirs(dataset_dir, exist_ok=True)
    
    dataset = EdgeColoringDataset(
        root=dataset_dir,
        graphs=all_graphs,
        colorings=all_colorings,
        processed_file_prefix='edge_coloring_data',
        seed=args.seed
    )
    
    # Split dataset
    train_idx, val_idx, test_idx = dataset.split(
        train_ratio=config.DATASET_PARAMS['train_ratio'],
        val_ratio=config.DATASET_PARAMS['val_ratio'],
        test_ratio=config.DATASET_PARAMS['test_ratio'],
        stratify_by=config.DATASET_PARAMS['stratify_by']
    )
    
    dataset.save_split(
        train_idx, val_idx, test_idx,
        os.path.join(dataset_dir, 'split_indices.json')
    )
    
    # Prepare tabular dataset for Random Forest models
    logger.info('Preparing tabular dataset...')
    tabular_dataset = TabularEdgeColoringDataset(
        graphs=all_graphs,
        colorings=all_colorings
    )
    
    # Extract features and prepare for model training
    X, y, feature_names = tabular_dataset.prepare_features(
        include_node_features=True,
        include_graph_features=True
    )
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = tabular_dataset.split(
        train_ratio=config.DATASET_PARAMS['train_ratio'],
        val_ratio=config.DATASET_PARAMS['val_ratio'],
        test_ratio=config.DATASET_PARAMS['test_ratio'],
        stratify=True,
        seed=args.seed
    )
    
    # Save tabular dataset
    tabular_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'tabular')
    os.makedirs(tabular_dir, exist_ok=True)
    
    np.savez(
        os.path.join(tabular_dir, 'train_data.npz'),
        X=X_train, y=y_train, feature_names=feature_names
    )
    np.savez(
        os.path.join(tabular_dir, 'val_data.npz'),
        X=X_val, y=y_val, feature_names=feature_names
    )
    np.savez(
        os.path.join(tabular_dir, 'test_data.npz'),
        X=X_test, y=y_test, feature_names=feature_names
    )
    
    logger.info(f'Prepared datasets with {len(all_graphs)} graph-coloring pairs')
    return dataset, (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)


@timer
def train_models(args):
    """Train machine learning models."""
    logger = logging.getLogger('model_training')
    logger.info('Training models...')
    
    # Training parameters
    epochs = args.epochs if args.epochs is not None else config.TRAINING_PARAMS['epochs']
    batch_size = args.batch_size if args.batch_size is not None else config.TRAINING_PARAMS['batch_size']
    lr = args.lr if args.lr is not None else config.TRAINING_PARAMS['lr']
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create directory for saving models
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Train models
    for model_type in args.models:
        logger.info(f'Training {model_type} model...')
        
        if model_type == 'random_forest':
            # Load tabular dataset
            tabular_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'tabular')
            train_data = np.load(os.path.join(tabular_dir, 'train_data.npz'), allow_pickle=True)
            val_data = np.load(os.path.join(tabular_dir, 'val_data.npz'), allow_pickle=True)
            
            X_train, y_train = train_data['X'], train_data['y']
            X_val, y_val = val_data['X'], val_data['y']
            feature_names = train_data['feature_names']
            
            # Initialize model
            rf_model = EdgeColoringRandomForest(
                mode='classification',
                n_estimators=config.MODEL_PARAMS['random_forest']['n_estimators'],
                max_depth=config.MODEL_PARAMS['random_forest']['max_depth'],
                min_samples_leaf=config.MODEL_PARAMS['random_forest']['min_samples_leaf'],
                random_state=config.MODEL_PARAMS['random_forest']['random_state']
            )
            
            # Train model
            logger.info('Training Random Forest model...')
            rf_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_accuracy = (rf_model.predict(X_val) == y_val).mean()
            logger.info(f'Validation accuracy: {val_accuracy:.4f}')
            
            # Save model
            rf_model.save_model(os.path.join(config.MODEL_DIR, 'random_forest.joblib'))
            
            # Save feature importance
            feature_importance = rf_model.get_feature_importance(feature_names)
            with open(os.path.join(config.MODEL_DIR, 'random_forest_feature_importance.json'), 'w') as f:
                json.dump(feature_importance, f)
            
        elif model_type == 'gnn':
            # Load PyTorch Geometric dataset
            dataset_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'pytorch_geometric')
            dataset = EdgeColoringDataset(root=dataset_dir)
            
            # Load splits
            train_idx, val_idx, _ = dataset.load_split(os.path.join(dataset_dir, 'split_indices.json'))
            
            # Initialize model
            gnn_model = GNNEdgeColoring(
                node_features=config.MODEL_PARAMS['gnn']['node_features'],
                edge_features=config.MODEL_PARAMS['gnn']['edge_features'],
                hidden_channels=config.MODEL_PARAMS['gnn']['hidden_channels'],
                num_layers=config.MODEL_PARAMS['gnn']['num_layers'],
                gnn_type=config.MODEL_PARAMS['gnn']['gnn_type'],
                dropout=config.MODEL_PARAMS['gnn']['dropout'],
                max_colors=config.MODEL_PARAMS['gnn']['max_colors']
            )
            
            # Train model
            logger.info('Training GNN model...')
            train_results = gnn_model.train(
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=config.TRAINING_PARAMS['weight_decay'],
                patience=config.TRAINING_PARAMS['patience'],
                device=device
            )
            
            # Save model
            gnn_model.save_model(os.path.join(config.MODEL_DIR, 'gnn_model.pt'))
            
            # Save training results
            with open(os.path.join(config.MODEL_DIR, 'gnn_training_results.json'), 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                results_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in train_results.items()}
                json.dump(results_dict, f)
            
        elif model_type == 'hybrid':
            # Initialize model
            hybrid_model = HybridEdgeColoring(
                base_model_type=config.MODEL_PARAMS['hybrid']['base_model_type'],
                ml_guidance_weight=config.MODEL_PARAMS['hybrid']['ml_guidance_weight'],
                local_search_iterations=config.MODEL_PARAMS['hybrid']['local_search_iterations'],
                random_seed=args.seed
            )
            
            # The hybrid model uses either Random Forest or GNN as its base model,
            # both of which we have already trained. So we just need to load them.
            if config.MODEL_PARAMS['hybrid']['base_model_type'] == 'random_forest':
                rf_model = EdgeColoringRandomForest()
                rf_model.load_model(os.path.join(config.MODEL_DIR, 'random_forest.joblib'))
                hybrid_model.ml_model = rf_model
            else:  # gnn
                gnn_model = GNNEdgeColoring()
                gnn_model.load_model(os.path.join(config.MODEL_DIR, 'gnn_model.pt'))
                hybrid_model.ml_model = gnn_model
            
            # Save hybrid model configuration
            with open(os.path.join(config.MODEL_DIR, 'hybrid_model_config.json'), 'w') as f:
                config_dict = {
                    'base_model_type': config.MODEL_PARAMS['hybrid']['base_model_type'],
                    'ml_guidance_weight': config.MODEL_PARAMS['hybrid']['ml_guidance_weight'],
                    'local_search_iterations': config.MODEL_PARAMS['hybrid']['local_search_iterations'],
                    'random_seed': args.seed
                }
                json.dump(config_dict, f)
    
    logger.info('Model training completed')


@timer
def evaluate_models(args):
    """Evaluate trained models."""
    logger = logging.getLogger('model_evaluation')
    logger.info('Evaluating models...')
    
    # Load test dataset
    logger.info('Loading test data...')
    
    # Tabular data for Random Forest
    tabular_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'tabular')
    test_data = np.load(os.path.join(tabular_dir, 'test_data.npz'), allow_pickle=True)
    X_test, y_test = test_data['X'], test_data['y']
    
    # PyTorch Geometric data for GNN
    dataset_dir = os.path.join(config.PROCESSED_DATA_DIR, 'datasets', 'pytorch_geometric')
    dataset = EdgeColoringDataset(root=dataset_dir)
    _, _, test_idx = dataset.load_split(os.path.join(dataset_dir, 'split_indices.json'))
    
    # Load raw graphs for baselines and hybrid model
    graphs = {}
    graph_dirs = [d for d in os.listdir(os.path.join(config.RAW_DATA_DIR, 'graphs')) 
                 if os.path.isdir(os.path.join(config.RAW_DATA_DIR, 'graphs', d))
                 and any(gt in d for gt in args.eval_graph_types)]
    
    for graph_dir in graph_dirs:
        graphs[graph_dir] = []
        dir_path = os.path.join(config.RAW_DATA_DIR, 'graphs', graph_dir)
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith('.graphml'):
                G = nx.read_graphml(os.path.join(dir_path, filename))
                # Convert node labels back to integers if needed
                if not isinstance(list(G.nodes())[0], int):
                    mapping = {n: int(n) for n in G.nodes()}
                    G = nx.relabel_nodes(G, mapping)
                graphs[graph_dir].append(G)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Results will be stored here
    results = {}
    
    # Evaluate baseline methods
    logger.info('Evaluating baseline methods...')
    for baseline in config.BASELINE_METHODS:
        logger.info(f'Evaluating {baseline}...')
        results[baseline] = {'color_count': [], 'time': [], 'graph_info': []}
        
        for graph_key, graph_list in graphs.items():
            for i, G in enumerate(graph_list):
                start_time = time.time()
                
                if baseline == 'random_ordering':
                    coloring = greedy_edge_coloring(G, 'random')
                elif baseline == 'degree_ordering':
                    coloring = greedy_edge_coloring(G, 'degree')
                elif baseline == 'vizing_implementation':
                    coloring = vizing_edge_coloring(G)
                elif baseline == 'tabu_search':
                    # Start with a greedy coloring and improve it
                    initial_coloring = greedy_edge_coloring(G, 'degree')
                    coloring = local_search_coloring(G, initial_coloring, method='tabu')
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # Calculate metrics
                num_colors = len(set(coloring.values()))
                max_degree = max(dict(G.degree()).values())
                color_count_ratio = num_colors / max_degree
                
                results[baseline]['color_count'].append(num_colors)
                results[baseline]['time'].append(computation_time)
                results[baseline]['graph_info'].append({
                    'graph_type': graph_key,
                    'graph_index': i,
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'max_degree': max_degree,
                    'color_count_ratio': color_count_ratio
                })
    
    # Evaluate ML models
    for model_type in args.models:
        logger.info(f'Evaluating {model_type} model...')
        results[model_type] = {'color_count': [], 'time': [], 'graph_info': []}
        
        if model_type == 'random_forest':
            # Load model
            rf_model = EdgeColoringRandomForest()
            rf_model.load_model(os.path.join(config.MODEL_DIR, 'random_forest.joblib'))
            
            # Evaluate on test set
            start_time = time.time()
            test_preds = rf_model.predict(X_test)
            end_time = time.time()
            
            # Calculate accuracy
            test_accuracy = (test_preds == y_test).mean()
            logger.info(f'Test accuracy: {test_accuracy:.4f}')
            
            # For a more comprehensive evaluation, we need to apply the RF model
            # to actual graph coloring tasks (implemented by the hybrid model below)
            
        elif model_type == 'gnn':
            # Load model
            gnn_model = GNNEdgeColoring()
            gnn_model.load_model(os.path.join(config.MODEL_DIR, 'gnn_model.pt'))
            
            # Evaluate on test set
            results_dict = gnn_model.evaluate(
                dataset=dataset,
                test_idx=test_idx,
                batch_size=config.EVALUATION_PARAMS['batch_size'],
                device=device
            )
            
            logger.info(f'Test accuracy: {results_dict["accuracy"]:.4f}')
            
            # For a more comprehensive evaluation, we need to apply the GNN model
            # to actual graph coloring tasks (implemented by the hybrid model below)
            
        elif model_type == 'hybrid':
            # Load hybrid model configuration
            with open(os.path.join(config.MODEL_DIR, 'hybrid_model_config.json'), 'r') as f:
                hybrid_config = json.load(f)
            
            # Initialize hybrid model
            hybrid_model = HybridEdgeColoring(
                base_model_type=hybrid_config['base_model_type'],
                ml_guidance_weight=hybrid_config['ml_guidance_weight'],
                local_search_iterations=hybrid_config['local_search_iterations'],
                random_seed=args.seed
            )
            
            # Load base ML model
            if hybrid_config['base_model_type'] == 'random_forest':
                rf_model = EdgeColoringRandomForest()
                rf_model.load_model(os.path.join(config.MODEL_DIR, 'random_forest.joblib'))
                hybrid_model.ml_model = rf_model
            else:  # gnn
                gnn_model = GNNEdgeColoring()
                gnn_model.load_model(os.path.join(config.MODEL_DIR, 'gnn_model.pt'))
                hybrid_model.ml_model = gnn_model
            
            # Evaluate on test graphs
            for graph_key, graph_list in graphs.items():
                for i, G in enumerate(graph_list):
                    start_time = time.time()
                    
                    # Generate coloring using hybrid approach
                    coloring = hybrid_model.color_graph(G)
                    
                    end_time = time.time()
                    computation_time = end_time - start_time
                    
                    # Calculate metrics
                    num_colors = len(set(coloring.values()))
                    max_degree = max(dict(G.degree()).values())
                    color_count_ratio = num_colors / max_degree
                    
                    results[model_type]['color_count'].append(num_colors)
                    results[model_type]['time'].append(computation_time)
                    results[model_type]['graph_info'].append({
                        'graph_type': graph_key,
                        'graph_index': i,
                        'num_nodes': G.number_of_nodes(),
                        'num_edges': G.number_of_edges(),
                        'max_degree': max_degree,
                        'color_count_ratio': color_count_ratio
                    })
    
    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_results(results, os.path.join(config.RESULTS_DIR, 'evaluation_results.json'))
    
    logger.info('Model evaluation completed')
    return results


@timer
def visualize_results(args, results=None):
    """Visualize evaluation results."""
    logger = logging.getLogger('visualization')
    logger.info('Visualizing results...')
    
    # Load results if not provided
    if results is None:
        with open(os.path.join(config.RESULTS_DIR, 'evaluation_results.json'), 'r') as f:
            results = json.load(f)
    
    # Create figures directory
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    
    # Comparative performance plot
    logger.info('Generating performance comparison plot...')
    fig1, ax1 = plot_performance_comparison(results, metric='color_count_ratio')
    fig1.savefig(os.path.join(config.FIGURE_DIR, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Feature importance plot (for Random Forest)
    if 'random_forest' in args.models:
        logger.info('Generating feature importance plot...')
        with open(os.path.join(config.MODEL_DIR, 'random_forest_feature_importance.json'), 'r') as f:
            feature_importance = json.load(f)
        
        fig2, ax2 = plot_feature_importance(feature_importance)
        fig2.savefig(os.path.join(config.FIGURE_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # Scaling behavior plot
    logger.info('Generating scaling behavior plot...')
    fig3, ax3 = plot_scaling_behavior(results)
    fig3.savefig(os.path.join(config.FIGURE_DIR, 'scaling_behavior.png'), dpi=300, bbox_inches='tight')
    
    # Solution quality plot
    logger.info('Generating solution quality plot...')
    fig4, ax4 = plot_solution_quality(results)
    fig4.savefig(os.path.join(config.FIGURE_DIR, 'solution_quality.png'), dpi=300, bbox_inches='tight')
    
    # Visualize a sample graph coloring
    if 'hybrid' in args.models:
        logger.info('Generating sample graph coloring visualization...')
        # Get a sample graph
        graph_dirs = [d for d in os.listdir(os.path.join(config.RAW_DATA_DIR, 'graphs')) 
                     if os.path.isdir(os.path.join(config.RAW_DATA_DIR, 'graphs', d))]
        
        sample_graph_dir = graph_dirs[0]
        dir_path = os.path.join(config.RAW_DATA_DIR, 'graphs', sample_graph_dir)
        G = nx.read_graphml(os.path.join(dir_path, os.listdir(dir_path)[0]))
        
        # Convert node labels back to integers if needed
        if not isinstance(list(G.nodes())[0], int):
            mapping = {n: int(n) for n in G.nodes()}
            G = nx.relabel_nodes(G, mapping)
        
        # Load hybrid model configuration
        with open(os.path.join(config.MODEL_DIR, 'hybrid_model_config.json'), 'r') as f:
            hybrid_config = json.load(f)
        
        # Initialize hybrid model
        hybrid_model = HybridEdgeColoring(
            base_model_type=hybrid_config['base_model_type'],
            ml_guidance_weight=hybrid_config['ml_guidance_weight'],
            local_search_iterations=hybrid_config['local_search_iterations'],
            random_seed=args.seed
        )
        
        # Load base ML model
        if hybrid_config['base_model_type'] == 'random_forest':
            rf_model = EdgeColoringRandomForest()
            rf_model.load_model(os.path.join(config.MODEL_DIR, 'random_forest.joblib'))
            hybrid_model.ml_model = rf_model
        else:  # gnn
            gnn_model = GNNEdgeColoring()
            gnn_model.load_model(os.path.join(config.MODEL_DIR, 'gnn_model.pt'))
            hybrid_model.ml_model = gnn_model
        
        # Generate coloring using hybrid approach
        coloring = hybrid_model.color_graph(G)
        
        # Visualize
        fig5, ax5 = visualize_graph_coloring(G, coloring)
        fig5.savefig(os.path.join(config.FIGURE_DIR, 'sample_coloring.png'), dpi=300, bbox_inches='tight')
    
    logger.info('Visualization completed')


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(config.LOGGING_CONFIG['filename'])
    logger.info('Starting Edge Coloring ML Framework')
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create necessary directories
    for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.RESULTS_DIR, config.MODEL_DIR, config.FIGURE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Execute workflow based on arguments
    if args.generate_graphs or args.run_all:
        graphs = generate_graph_instances(args)
    else:
        graphs = None
    
    if args.generate_colorings or args.run_all:
        colorings = generate_edge_colorings(args, graphs)
    else:
        colorings = None
    
    if args.extract_features or args.run_all:
        graph_features, edge_features = extract_features_for_graphs(args, graphs, colorings)
    else:
        graph_features, edge_features = None, None
    
    if args.prepare_datasets or args.run_all:
        dataset, tabular_data = prepare_datasets(args, graphs, colorings, graph_features, edge_features)
    
    if args.train_models or args.run_all:
        train_models(args)
    
    if args.evaluate_models or args.run_all:
        results = evaluate_models(args)
    else:
        results = None
    
    if args.visualize_results or args.run_all:
        visualize_results(args, results)
    
    logger.info('Edge Coloring ML Framework completed')


if __name__ == '__main__':
    main()