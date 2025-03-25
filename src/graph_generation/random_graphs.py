# src/graph_generation/random_graphs.py
import networkx as nx
import numpy as np

def generate_random_graph(n, p, connected=True, max_attempts=100):
    """
    Generate a random Erdős-Rényi graph
    
    Parameters:
    -----------
    n : int
        Number of vertices
    p : float
        Probability of edge creation
    connected : bool
        Whether to ensure the graph is connected
    max_attempts : int
        Maximum number of attempts to generate a connected graph
        
    Returns:
    --------
    G : networkx.Graph
        Random graph
    """
    attempts = 0
    while attempts < max_attempts:
        G = nx.erdos_renyi_graph(n, p)
        
        if not connected or nx.is_connected(G):
            # Add edge indices as attributes
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['index'] = i
            return G
        
        attempts += 1
    
    # If we couldn't generate a connected graph, take the last one
    # and add edges to make it connected
    components = list(nx.connected_components(G))
    
    for i in range(len(components) - 1):
        u = np.random.choice(list(components[i]))
        v = np.random.choice(list(components[i + 1]))
        G.add_edge(u, v)
    
    # Add edge indices
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['index'] = i
        
    return G