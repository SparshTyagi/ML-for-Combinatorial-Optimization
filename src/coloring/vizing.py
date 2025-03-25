# src/coloring/vizing.py
import networkx as nx
import random

def vizing_edge_coloring(graph):
    """
    Implementation of Vizing's algorithm for edge coloring
    This is a heuristic implementation that aims to color with Δ or Δ+1 colors
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
        
    Returns:
    --------
    coloring : dict
        Dictionary mapping edge -> color
    """
    # Get maximum degree
    max_degree = max(dict(graph.degree()).values())
    
    # Initialize available colors as 0 to max_degree + 1
    # (Using 0 as a special "no color" value)
    available_colors = list(range(1, max_degree + 2))
    
    # Initialize coloring
    edges = list(graph.edges())
    # Canonicalize edge representation (smaller node id first)
    edges = [tuple(sorted(edge)) for edge in edges]
    
    coloring = {}
    
    # Helper function to get colored edges incident to a vertex
    def colored_edges(v):
        result = []
        for e in graph.edges(v):
            e = tuple(sorted(e))
            if e in coloring:
                result.append(e)
        return result
    
    # Helper function to get colors used by edges incident to a vertex
    def used_colors(v):
        result = set()
        for e in colored_edges(v):
            result.add(coloring[e])
        return result
    
    # Helper function to find an available color for an edge
    def find_available_color(u, v):
        used_u = used_colors(u)
        used_v = used_colors(v)
        for color in available_colors:
            if color not in used_u and color not in used_v:
                return color
        return None
    
    # Helper function to find a fan
    def find_fan(u, v, color1, color2):
        fan_vertices = [v]
        fan_colors = [color1]
        
        while True:
            # Find a vertex w such that edge (u,w) is colored with fan_colors[-1]
            # and w is not in fan_vertices yet
            found = False
            for w in graph.neighbors(u):
                edge = tuple(sorted([u, w]))
                if edge in coloring and coloring[edge] == fan_colors[-1] and w not in fan_vertices:
                    fan_vertices.append(w)
                    # Find a color missing at w but present at v
                    for c in available_colors:
                        if c not in used_colors(w) and c in used_colors(fan_vertices[0]) and c != color2:
                            fan_colors.append(c)
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                break
        
        return fan_vertices, fan_colors
    
    # Helper function to rotate a fan
    def rotate_fan(u, fan_vertices, fan_colors):
        for i in range(len(fan_vertices) - 1):
            v = fan_vertices[i]
            w = fan_vertices[i + 1]
            edge_uv = tuple(sorted([u, v]))
            edge_uw = tuple(sorted([u, w]))
            
            coloring[edge_uv] = fan_colors[i + 1]
    
    # First pass: try to color all edges with the greedy algorithm
    for edge in edges:
        color = find_available_color(edge[0], edge[1])
        if color is not None:
            coloring[edge] = color
    
    # Second pass: try to color remaining edges with Vizing's algorithm
    uncolored_edges = [e for e in edges if e not in coloring]
    
    for edge in uncolored_edges:
        u, v = edge
        
        # Find colors missing at u and v
        missing_u = set(available_colors) - used_colors(u)
        missing_v = set(available_colors) - used_colors(v)
        
        if missing_u and missing_v:
            # If there's a color missing at both endpoints, use it
            color = min(missing_u.intersection(missing_v))
            coloring[edge] = color
        else:
            # Choose a color missing at u (must exist since |used_colors(u)| ≤ deg(u) < |available_colors|)
            color1 = min(missing_u)
            
            # Find a color present at v but missing at u
            color2 = None
            for c in used_colors(v):
                if c not in used_colors(u):
                    color2 = c
                    break
            
            if color2 is not None:
                # Find fan
                fan_vertices, fan_colors = find_fan(u, v, color2, color1)
                
                if len(fan_vertices) > 1:
                    # Rotate fan
                    rotate_fan(u, fan_vertices, fan_colors)
                
                # Color the edge with color1
                coloring[edge] = color1
            else:
                # Fallback: just use a new color
                coloring[edge] = max(coloring.values()) + 1 if coloring else 1
    
    return coloring