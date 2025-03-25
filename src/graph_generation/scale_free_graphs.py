# src/graph_generation/scale_free_graphs.py
import networkx as nx

def generate_scale_free_graph(n, m, seed=None):
    """
    Generate a scale-free graph using the Barabási-Albert model
    
    Parameters:
    -----------
    n : int
        Number of vertices
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    G : networkx.Graph
        Scale-free graph
    """
    # Ensure we can create a valid BA graph
    m = min(m, n-1)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    
    # Add edge indices as attributes
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['index'] = i
        
    return G