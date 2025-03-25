# src/coloring/__init__.py
import time
from src.coloring.greedy import (
    greedy_edge_coloring_random,
    greedy_edge_coloring_degree_based,
    greedy_edge_coloring_centrality_based
)
from src.coloring.vizing import vizing_edge_coloring
from src.coloring.ilp import ilp_edge_coloring
from src.coloring.local_search import local_search_coloring

def get_lower_bound(graph):
    """Get theoretical lower bound (maximum degree) for edge coloring"""
    return max(dict(graph.degree()).values())

def validate_coloring(graph, coloring):
    """
    Validate if the coloring is valid (no adjacent edges have same color)
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    coloring : dict
        Dictionary mapping edge -> color
        
    Returns:
    --------
    valid : bool
        True if coloring is valid, False otherwise
    conflicts : list
        List of conflicting edge pairs
    """
    valid = True
    conflicts = []
    
    # Create a dictionary to store edges by color
    edges_by_color = {}
    for edge, color in coloring.items():
        if color not in edges_by_color:
            edges_by_color[color] = []
        edges_by_color[color].append(edge)
    
    # Check for conflicts
    for color, edges in edges_by_color.items():
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                e1 = edges[i]
                e2 = edges[j]
                # Check if edges share a vertex
                if set(e1).intersection(set(e2)):
                    valid = False
                    conflicts.append((e1, e2))
    
    return valid, conflicts

def color_count_ratio(graph, coloring):
    """
    Calculate the ratio of colors used to the theoretical lower bound
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    coloring : dict
        Dictionary mapping edge -> color
        
    Returns:
    --------
    ratio : float
        Ratio of colors used to maximum degree
    """
    max_degree = max(dict(graph.degree()).values())
    num_colors = len(set(coloring.values()))
    return num_colors / max_degree

def generate_all_colorings(graph):
    """
    Generate colorings using all available methods
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
        
    Returns:
    --------
    colorings : dict
        Dictionary mapping method name to (coloring, stats) tuple
    """
    colorings = {}
    
    # Only use ILP for small graphs
    if graph.number_of_nodes() <= 50:
        try:
            start_time = time.time()
            ilp_coloring = ilp_edge_coloring(graph)
            ilp_time = time.time() - start_time
            valid, conflicts = validate_coloring(graph, ilp_coloring)
            
            colorings['ilp'] = {
                'coloring': ilp_coloring,
                'num_colors': len(set(ilp_coloring.values())),
                'time': ilp_time,
                'valid': valid,
                'conflicts': len(conflicts),
                'ratio': color_count_ratio(graph, ilp_coloring)
            }
        except Exception as e:
            print(f"ILP coloring failed: {e}")
    
    # Greedy with random ordering
    start_time = time.time()
    random_coloring = greedy_edge_coloring_random(graph)
    random_time = time.time() - start_time
    valid, conflicts = validate_coloring(graph, random_coloring)
    
    colorings['random'] = {
        'coloring': random_coloring,
        'num_colors': len(set(random_coloring.values())),
        'time': random_time,
        'valid': valid,
        'conflicts': len(conflicts),
        'ratio': color_count_ratio(graph, random_coloring)
    }
    
    # Greedy with degree-based ordering
    start_time = time.time()
    degree_coloring = greedy_edge_coloring_degree_based(graph)
    degree_time = time.time() - start_time
    valid, conflicts = validate_coloring(graph, degree_coloring)
    
    colorings['degree'] = {
        'coloring': degree_coloring,
        'num_colors': len(set(degree_coloring.values())),
        'time': degree_time,
        'valid': valid,
        'conflicts': len(conflicts),
        'ratio': color_count_ratio(graph, degree_coloring)
    }
    
    # Greedy with centrality-based ordering
    start_time = time.time()
    centrality_coloring = greedy_edge_coloring_centrality_based(graph)
    centrality_time = time.time() - start_time
    valid, conflicts = validate_coloring(graph, centrality_coloring)
    
    colorings['centrality'] = {
        'coloring': centrality_coloring,
        'num_colors': len(set(centrality_coloring.values())),
        'time': centrality_time,
        'valid': valid,
        'conflicts': len(conflicts),
        'ratio': color_count_ratio(graph, centrality_coloring)
    }
    
    # Vizing's algorithm
    start_time = time.time()
    vizing_coloring = vizing_edge_coloring(graph)
    vizing_time = time.time() - start_time
    valid, conflicts = validate_coloring(graph, vizing_coloring)
    
    colorings['vizing'] = {
        'coloring': vizing_coloring,
        'num_colors': len(set(vizing_coloring.values())),
        'time': vizing_time,
        'valid': valid,
        'conflicts': len(conflicts),
        'ratio': color_count_ratio(graph, vizing_coloring)
    }
    
    # Local search improvement on the best greedy coloring
    best_greedy = min(
        [colorings['random'], colorings['degree'], colorings['centrality']],
        key=lambda x: x['num_colors']
    )
    
    start_time = time.time()
    local_search_coloring_result = local_search_coloring(graph, best_greedy['coloring'])
    local_search_time = time.time() - start_time
    valid, conflicts = validate_coloring(graph, local_search_coloring_result)
    
    colorings['local_search'] = {
        'coloring': local_search_coloring_result,
        'num_colors': len(set(local_search_coloring_result.values())),
        'time': local_search_time + best_greedy['time'],  # Include time for initial coloring
        'valid': valid,
        'conflicts': len(conflicts),
        'ratio': color_count_ratio(graph, local_search_coloring_result)
    }
    
    return colorings