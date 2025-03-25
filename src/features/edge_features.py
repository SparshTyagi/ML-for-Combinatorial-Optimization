# src/features/edge_features.py
import networkx as nx
import numpy as np
import warnings

def extract_edge_features(graph):
    """
    Extract features for each edge in the graph
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
        
    Returns:
    --------
    edge_features : dict
        Dictionary mapping edges to their feature vectors
    """
    edge_features = {}
    
    # Cache degree information
    degrees = dict(graph.degree())
    
    # Cache clustering coefficients (for smaller graphs)
    if graph.number_of_nodes() <= 1000:
        try:
            clustering = nx.clustering(graph)
        except Exception:
            clustering = {node: 0 for node in graph.nodes()}
    else:
        clustering = {node: 0 for node in graph.nodes()}
    
    # Try to compute edge betweenness centrality (for smaller graphs)
    if graph.number_of_nodes() <= 500:
        try:
            edge_betweenness = nx.edge_betweenness_centrality(graph)
        except Exception:
            edge_betweenness = {e: 0 for e in graph.edges()}
    else:
        edge_betweenness = {e: 0 for e in graph.edges()}
    
    # Process each edge
    for u, v in graph.edges():
        # Canonicalize edge representation (smaller node id first)
        edge = tuple(sorted([u, v]))
        
        # Initialize feature dictionary for this edge
        features = {}
        
        # Endpoint properties
        features['degree_u'] = degrees[u]
        features['degree_v'] = degrees[v]
        features['degree_sum'] = degrees[u] + degrees[v]
        features['degree_product'] = degrees[u] * degrees[v]
        features['degree_max'] = max(degrees[u], degrees[v])
        features['degree_min'] = min(degrees[u], degrees[v])
        features['degree_diff'] = abs(degrees[u] - degrees[v])
        
        features['clustering_u'] = clustering[u]
        features['clustering_v'] = clustering[v]
        features['clustering_avg'] = (clustering[u] + clustering[v]) / 2
        
        # Neighborhood structure
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        common_neighbors = neighbors_u.intersection(neighbors_v)
        
        features['common_neighbors'] = len(common_neighbors)
        features['total_neighbors'] = len(neighbors_u.union(neighbors_v)) - 2  # -2 to exclude u and v themselves
        features['jaccard_coefficient'] = len(common_neighbors) / max(1, len(neighbors_u.union(neighbors_v)) - 2)
        
        # Number of adjacent edges
        features['num_adjacent_edges'] = degrees[u] + degrees[v] - 2  # -2 to exclude the edge itself
        
        # Edge betweenness centrality
        features['betweenness'] = edge_betweenness.get((u, v), 0) or edge_betweenness.get((v, u), 0)
        
        # Triangle participation
        if graph.number_of_nodes() <= 1000:
            try:
                triangles = sum(1 for n in common_neighbors)
                features['triangle_count'] = triangles
            except Exception:
                features['triangle_count'] = 0
        else:
            features['triangle_count'] = 0
        
        # Conflict potential (estimate of how many edges this edge might conflict with)
        features['conflict_potential'] = features['num_adjacent_edges']
        
        # Store features for this edge
        edge_features[edge] = features
    
    return edge_features