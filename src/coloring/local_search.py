# src/coloring/local_search.py
import networkx as nx
import random
import copy
import time

def local_search_coloring(graph, initial_coloring, max_iterations=1000, tabu_size=10):
    """
    Local search for edge coloring using tabu search
    
    Parameters:
    -----------
    graph : networkx.Graph
        The input graph
    initial_coloring : dict
        Initial edge coloring
    max_iterations : int
        Maximum number of iterations
    tabu_size : int
        Size of the tabu list
        
    Returns:
    --------
    best_coloring : dict
        Best coloring found
    """
    # Helper function to calculate the number of colors used
    def num_colors(coloring):
        return len(set(coloring.values()))
    
    # Helper function to check if an edge can be recolored with a new color
    def can_recolor(edge, new_color, coloring):
        adjacent_edges = []
        u, v = edge
        
        for w in graph.neighbors(u):
            if w != v:
                adjacent_edges.append((u, w) if u < w else (w, u))
        
        for w in graph.neighbors(v):
            if w != u:
                adjacent_edges.append((v, w) if v < w else (w, v))
        
        return all(coloring.get(adj_edge) != new_color for adj_edge in adjacent_edges)
    
    # Start with the initial coloring
    current_coloring = copy.deepcopy(initial_coloring)
    best_coloring = copy.deepcopy(initial_coloring)
    best_num_colors = num_colors(best_coloring)
    
    # Initialize tabu list
    tabu_list = []
    
    # Main tabu search loop
    for iteration in range(max_iterations):
        # Get the current number of colors and find candidate moves
        current_num_colors = num_colors(current_coloring)
        moves = []
        
        # Consider all edges
        for edge in initial_coloring.keys():
            current_color = current_coloring[edge]
            
            # Try to recolor with a lower color
            for new_color in range(1, current_num_colors):
                if new_color != current_color and can_recolor(edge, new_color, current_coloring):
                    # Calculate the objective value change
                    new_coloring = copy.deepcopy(current_coloring)
                    new_coloring[edge] = new_color
                    new_num_colors = num_colors(new_coloring)
                    
                    # Add to candidate moves
                    moves.append((edge, new_color, new_num_colors))
        
        # If no valid moves, try to make a small random perturbation
        if not moves:
            edges = list(initial_coloring.keys())
            random_edge = random.choice(edges)
            max_color = max(current_coloring.values())
            
            # Try a random new color
            new_color = random.randint(1, max_color + 1)
            if can_recolor(random_edge, new_color, current_coloring):
                current_coloring[random_edge] = new_color
            continue
        
        # Sort moves by number of colors (ascending)
        moves.sort(key=lambda x: x[2])
        
        # Find the best non-tabu move
        selected_move = None
        for move in moves:
            edge, new_color, _ = move
            if (edge, new_color) not in tabu_list:
                selected_move = move
                break
        
        # If no non-tabu move, select the best move anyway (aspiration criterion)
        if selected_move is None:
            selected_move = moves[0]
        
        # Apply the selected move
        edge, new_color, new_num_colors = selected_move
        current_coloring[edge] = new_color
        
        # Update the tabu list
        tabu_list.append((edge, new_color))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        
        # Update best solution if improved
        if new_num_colors < best_num_colors:
            best_coloring = copy.deepcopy(current_coloring)
            best_num_colors = new_num_colors
    
    return best_coloring