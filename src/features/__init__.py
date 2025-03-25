# src/features/__init__.py
from src.features.graph_features import extract_graph_features
from src.features.edge_features import extract_edge_features

def extract_all_features(graph):
    """
    Extract all graph and edge features
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
        
    Returns:
    --------
    features : dict
        Dictionary with 'graph' and 'edge' features
    """
    graph_features = extract_graph_features(graph)
    edge_features = extract_edge_features(graph)
    
    return {
        'graph': graph_features,
        'edge': edge_features
    }