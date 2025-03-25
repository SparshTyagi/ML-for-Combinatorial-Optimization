# src/coloring/greedy.py
import networkx as nx
import random
import time

def order_edges(graph, strategy='random'):
    """
    Order edges according to the specified strategy
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    strategy : str
        Edge ordering strategy ('random', 'degree', 'centrality')
        
    Returns:
    --------
    ordered_edges : list
        List of edges in the desired order
    """
    edges = list(graph.edges())
    
    if strategy == 'random':
        random.shuffle(edges)
        return edges
    
    elif strategy == 'degree':
        # Order by sum of endpoint degrees (descending)
        edge_degrees = [(u, v, graph.degree(u) + graph.degree(v)) for u, v in edges]
        return [edge[:2] for edge in sorted(edge_degrees, key=lambda x: x[2], reverse=True)]
    
    elif strategy == 'centrality':
        # Order by edge betweenness centrality (descending)
        edge_centrality = nx.edge_betweenness_centrality(graph)
        edge_centrality_list = [(u, v, edge_centrality[(u, v)]) for u, v in edges]
        return [edge[:2] for edge in sorted(edge_centrality_list, key=lambda x: x[2], reverse=True)]
    
    else:
        raise ValueError(f"Unknown edge ordering strategy: {strategy}")

def get_adjacent_edges(graph, edge):
    """
    Get all edges adjacent to the given edge
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    edge : tuple
        The edge (u, v)
        
    Returns:
    --------
    adjacent_edges : list
        List of adjacent edges
    """
    u, v = edge
    adjacent_edges = []
    
    for w in graph.neighbors(u):
        if w != v:
            adjacent_edges.append((u, w) if u < w else (w, u))
    
    for w in graph.neighbors(v):
        if w != u:
            adjacent_edges.append((v, w) if v < w else (w, v))
    
    return adjacent_edges

def greedy_edge_coloring(graph, edge_ordering_strategy='random'):
    """
    Greedy edge coloring algorithm
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    edge_ordering_strategy : str
        Strategy for ordering edges ('random', 'degree', 'centrality')
        
    Returns:
    --------
    coloring : dict
        Dictionary mapping edge -> color
    """
    # Order edges according to the specified strategy
    ordered_edges = order_edges(graph, strategy=edge_ordering_strategy)
    
    # Initialize colors
    coloring = {}
    
    # Assign colors greedily
    for edge in ordered_edges:
        # Canonicalize edge representation (smaller node id first)
        edge = tuple(sorted(edge))
        
        # Find the first available color
        available_colors = set(range(1, graph.number_of_edges() + 1))
        
        for neighbor in get_adjacent_edges(graph, edge):
            neighbor = tuple(sorted(neighbor))
            if neighbor in coloring:
                available_colors.discard(coloring[neighbor])
        
        # Assign the smallest available color
        coloring[edge] = min(available_colors)
    
    return coloring

def greedy_edge_coloring_random(graph):
    """Greedy edge coloring with random edge ordering"""
    return greedy_edge_coloring(graph, edge_ordering_strategy='random')

def greedy_edge_coloring_degree_based(graph):
    """Greedy edge coloring with degree-based edge ordering"""
    return greedy_edge_coloring(graph, edge_ordering_strategy='degree')

def greedy_edge_coloring_centrality_based(graph):
    """Greedy edge coloring with centrality-based edge ordering"""
    return greedy_edge_coloring(graph, edge_ordering_strategy='centrality')