from src.models.random_forest import EdgeColoringRandomForest
from src.models.gnn import GNNEdgeColoring
from src.models.hybrid_model import HybridEdgeColoring

def train_all_models(data, train_indices, val_indices, task='edge_classification', **kwargs):
    """
    Train all model types on the given data.
    
    Args:
        data: Dataset (format depends on model type)
        train_indices: Indices for training
        val_indices: Indices for validation
        task: 'edge_classification' or 'graph_regression'
        **kwargs: Additional training arguments
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # Train Random Forest model
    rf_model = EdgeColoringRandomForest()
    rf_model.train(data, train_indices, val_indices, task=task, **kwargs)
    models['random_forest'] = rf_model
    
    # Train GNN model
    gnn_model = GNNEdgeColoring()
    gnn_model.train(data, train_indices, val_indices, task=task, **kwargs)
    models['gnn'] = gnn_model
    
    # Train Hybrid model
    hybrid_model = HybridEdgeColoring(base_model_type='gnn')
    hybrid_model.train(data, train_indices, val_indices, task=task, **kwargs)
    models['hybrid'] = hybrid_model
    
    return models