# src/graph_generation/small_world_graphs.py
import networkx as nx

def generate_small_world_graph(n, k, p, seed=None):
    """
    Generate a small-world graph using the Watts-Strogatz model
    
    Parameters:
    -----------
    n : int
        Number of vertices
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    G : networkx.Graph
        Small-world graph
    """
    # Ensure k is even as required by Watts-Strogatz model
    if k % 2 == 1:
        k += 1
    
    # Ensure k < n
    k = min(k, n-1)
    
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    
    # Add edge indices as attributes
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['index'] = i
        
    return G