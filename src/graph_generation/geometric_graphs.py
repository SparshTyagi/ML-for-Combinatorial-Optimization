# src/graph_generation/geometric_graphs.py
import networkx as nx
import numpy as np

def generate_geometric_graph(n, radius, seed=None, ensure_connected=True):
    """
    Generate a random geometric graph
    
    Parameters:
    -----------
    n : int
        Number of vertices
    radius : float
        Distance threshold for connecting vertices
    seed : int, optional
        Random seed for reproducibility
    ensure_connected : bool
        Whether to ensure the graph is connected
        
    Returns:
    --------
    G : networkx.Graph
        Geometric graph
    """
    if seed is not None:
        np.random.seed(seed)
        
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts:
        G = nx.random_geometric_graph(n, radius, seed=seed)
        
        if not ensure_connected or nx.is_connected(G):
            # Add edge indices as attributes
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['index'] = i
            
            # Add positions as node attributes and remove the original pos attribute
            pos = nx.get_node_attributes(G, 'pos')
            for node, position in pos.items():
                G.nodes[node]['x'] = float(position[0])
                G.nodes[node]['y'] = float(position[1])
                # Remove the original pos attribute to avoid GraphML serialization issues
                del G.nodes[node]['pos']
                
            return G
        
        attempts += 1
        # Generate a different graph in the next attempt
        if seed is not None:
            seed += 1
        radius *= 1.05  # Slightly increase radius to increase connectivity
    
    # If still not connected, add edges to connect components
    components = list(nx.connected_components(G))
    
    for i in range(len(components) - 1):
        u = np.random.choice(list(components[i]))
        v = np.random.choice(list(components[i + 1]))
        G.add_edge(u, v)
    
    # Add edge indices
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['index'] = i
        
    # Add positions as node attributes and remove the original pos attribute
    pos = nx.get_node_attributes(G, 'pos')
    for node, position in pos.items():
        G.nodes[node]['x'] = float(position[0])
        G.nodes[node]['y'] = float(position[1])
        # Remove the original pos attribute to avoid GraphML serialization issues
        del G.nodes[node]['pos']
        
    return G