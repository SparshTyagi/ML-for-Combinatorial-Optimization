# src/features/graph_features.py
import networkx as nx
import numpy as np
from scipy import stats
import warnings

def extract_graph_features(graph):
    """
    Extract graph-level features
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
        
    Returns:
    --------
    features : dict
        Dictionary of graph features
    """
    features = {}
    
    # Basic properties
    features['num_nodes'] = graph.number_of_nodes()
    features['num_edges'] = graph.number_of_edges()
    features['density'] = nx.density(graph)
    
    # Degree statistics
    degrees = dict(graph.degree()).values()
    features['max_degree'] = max(degrees)
    features['min_degree'] = min(degrees)
    features['avg_degree'] = sum(degrees) / len(degrees)
    features['degree_variance'] = np.var(list(degrees))
    features['degree_skewness'] = stats.skew(list(degrees))
    
    # Spectral properties
    try:
        # Compute adjacency matrix eigenvalues
        eigenvalues = np.linalg.eigvalsh(nx.adjacency_matrix(graph).toarray())
        features['spectral_radius'] = max(abs(eigenvalues))
        features['spectral_gap'] = abs(eigenvalues[-1] - eigenvalues[-2]) if len(eigenvalues) > 1 else 0
        features['energy'] = sum(abs(eigenvalues))
        
        # Compute Laplacian eigenvalues
        laplacian_eigenvalues = np.linalg.eigvalsh(nx.laplacian_matrix(graph).toarray())
        features['algebraic_connectivity'] = laplacian_eigenvalues[1] if len(laplacian_eigenvalues) > 1 else 0
    except Exception as e:
        # For large graphs, compute approximate spectral properties
        warnings.warn(f"Computing approximate spectral properties due to: {e}")
        features['spectral_radius'] = max(degrees)  # Upper bound
        features['spectral_gap'] = 0
        features['energy'] = 2 * graph.number_of_edges()  # Approximation
        features['algebraic_connectivity'] = 0
    
    # Structural properties
    if graph.number_of_nodes() <= 1000:  # Only compute for reasonably sized graphs
        try:
            features['clustering_coefficient'] = nx.average_clustering(graph)
            
            if nx.is_connected(graph):
                features['diameter'] = nx.diameter(graph)
                features['avg_path_length'] = nx.average_shortest_path_length(graph)
            else:
                # For disconnected graphs, compute properties on the largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                largest_cc_graph = graph.subgraph(largest_cc)
                features['diameter'] = nx.diameter(largest_cc_graph)
                features['avg_path_length'] = nx.average_shortest_path_length(largest_cc_graph)
                
            features['num_triangles'] = sum(nx.triangles(graph).values()) / 3
        except Exception as e:
            # For very large graphs, use approximations or skip some metrics
            warnings.warn(f"Using approximations for structural properties due to: {e}")
            features['clustering_coefficient'] = 0
            features['diameter'] = 0
            features['avg_path_length'] = 0
            features['num_triangles'] = 0
    else:
        # For very large graphs, use approximations
        features['clustering_coefficient'] = 0
        features['diameter'] = 0
        features['avg_path_length'] = 0
        features['num_triangles'] = 0
    
    # Try to compute assortativity
    try:
        features['assortativity'] = nx.degree_assortativity_coefficient(graph)
    except Exception:
        features['assortativity'] = 0
    
    # Centrality measures (for smaller graphs)
    if graph.number_of_nodes() <= 500:
        try:
            # Compute centrality metrics
            degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
            features['avg_degree_centrality'] = np.mean(degree_centrality)
            features['std_degree_centrality'] = np.std(degree_centrality)
            
            betweenness_centrality = np.array(list(nx.betweenness_centrality(graph).values()))
            features['avg_betweenness_centrality'] = np.mean(betweenness_centrality)
            features['std_betweenness_centrality'] = np.std(betweenness_centrality)
            
            if graph.number_of_nodes() <= 200:  # Eigenvector centrality is expensive
                eigenvector_centrality = np.array(list(nx.eigenvector_centrality_numpy(graph).values()))
                features['avg_eigenvector_centrality'] = np.mean(eigenvector_centrality)
                features['std_eigenvector_centrality'] = np.std(eigenvector_centrality)
            else:
                features['avg_eigenvector_centrality'] = 0
                features['std_eigenvector_centrality'] = 0
        except Exception as e:
            # For problematic graphs, use default values
            warnings.warn(f"Using default values for centrality due to: {e}")
            features['avg_degree_centrality'] = 0
            features['std_degree_centrality'] = 0
            features['avg_betweenness_centrality'] = 0
            features['std_betweenness_centrality'] = 0
            features['avg_eigenvector_centrality'] = 0
            features['std_eigenvector_centrality'] = 0
    else:
        # For very large graphs, skip centrality computations
        features['avg_degree_centrality'] = 0
        features['std_degree_centrality'] = 0
        features['avg_betweenness_centrality'] = 0
        features['std_betweenness_centrality'] = 0
        features['avg_eigenvector_centrality'] = 0
        features['std_eigenvector_centrality'] = 0
    
    return features