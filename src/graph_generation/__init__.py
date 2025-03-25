# src/graph_generation/__init__.py
from src.graph_generation.random_graphs import generate_random_graph
from src.graph_generation.scale_free_graphs import generate_scale_free_graph
from src.graph_generation.small_world_graphs import generate_small_world_graph
from src.graph_generation.geometric_graphs import generate_geometric_graph

def generate_graphs(sizes, graph_types, params, num_per_config=5):
    """
    Generate a collection of graphs with different sizes and types
    
    Parameters:
    -----------
    sizes : list
        List of graph sizes (number of vertices)
    graph_types : list
        List of graph types to generate ('random', 'scale_free', 'small_world', 'geometric')
    params : dict
        Dictionary with parameters for each graph type
    num_per_config : int
        Number of graphs to generate per configuration
        
    Returns:
    --------
    graphs : list
        List of (graph, metadata) tuples
    """
    graphs = []
    
    for size in sizes:
        if 'random' in graph_types:
            for p in params.get('random_p', [0.5]):
                for i in range(num_per_config):
                    g = generate_random_graph(size, p)
                    metadata = {
                        'type': 'random',
                        'n': size,
                        'p': p,
                        'instance': i
                    }
                    graphs.append((g, metadata))
        
        if 'scale_free' in graph_types:
            for m in params.get('scale_free_m', [2]):
                for i in range(num_per_config):
                    g = generate_scale_free_graph(size, m)
                    metadata = {
                        'type': 'scale_free',
                        'n': size,
                        'm': m,
                        'instance': i
                    }
                    graphs.append((g, metadata))
        
        if 'small_world' in graph_types:
            for k in params.get('small_world_k', [4]):
                for p in params.get('small_world_p', [0.1]):
                    for i in range(num_per_config):
                        g = generate_small_world_graph(size, k, p)
                        metadata = {
                            'type': 'small_world',
                            'n': size,
                            'k': k,
                            'p': p,
                            'instance': i
                        }
                        graphs.append((g, metadata))
        
        if 'geometric' in graph_types:
            for r in params.get('geometric_r', [0.2]):
                for i in range(num_per_config):
                    g = generate_geometric_graph(size, r)
                    metadata = {
                        'type': 'geometric',
                        'n': size,
                        'r': r,
                        'instance': i
                    }
                    graphs.append((g, metadata))
    
    return graphs