# src/coloring/ilp.py
import networkx as nx
import numpy as np
import pulp
import time

def ilp_edge_coloring(graph, time_limit=300):
    """
    Edge coloring using Integer Linear Programming
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    time_limit : int
        Time limit for the solver in seconds
        
    Returns:
    --------
    coloring : dict
        Dictionary mapping edge -> color
    """
    # Get maximum degree (theoretical lower bound)
    max_degree = max(dict(graph.degree()).values())
    
    # Number of potential colors (max_degree + 1 by Vizing's theorem)
    n_colors = max_degree + 1
    
    # Initialize the model
    model = pulp.LpProblem("EdgeColoring", pulp.LpMinimize)
    
    # Create edge list with fixed order
    edges = list(graph.edges())
    edges = [tuple(sorted(e)) for e in edges]
    
    # Create variables
    # x[e][c] = 1 if edge e is assigned color c, 0 otherwise
    x = {(e, c): pulp.LpVariable(f"x_{e}_{c}", cat=pulp.LpBinary) 
         for e in edges for c in range(1, n_colors + 1)}
    
    # y[c] = 1 if color c is used, 0 otherwise
    y = {c: pulp.LpVariable(f"y_{c}", cat=pulp.LpBinary) 
         for c in range(1, n_colors + 1)}
    
    # Objective: minimize the number of colors used
    model += pulp.lpSum(y[c] for c in range(1, n_colors + 1))
    
    # Constraint: each edge is assigned exactly one color
    for e in edges:
        model += pulp.lpSum(x[(e, c)] for c in range(1, n_colors + 1)) == 1
    
    # Constraint: adjacent edges have different colors
    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges[i+1:], i+1):
            # Check if edges are adjacent
            if set(e1).intersection(set(e2)):
                for c in range(1, n_colors + 1):
                    model += x[(e1, c)] + x[(e2, c)] <= 1
    
    # Constraint: link edge color assignment to color usage
    for c in range(1, n_colors + 1):
        for e in edges:
            model += x[(e, c)] <= y[c]
    
    # Set time limit
    model.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
    
    # Extract solution
    coloring = {}
    for e in edges:
        for c in range(1, n_colors + 1):
            if pulp.value(x[(e, c)]) > 0.5:  # Binary variable might have small numerical errors
                coloring[e] = c
                break
    
    return coloring