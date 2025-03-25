import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

def visualize_graph(G, title="Graph Visualization", layout='spring', node_size=300, 
                   node_color='skyblue', figsize=(10, 8), save_path=None):
    """
    Visualize a graph.
    
    Args:
        G (networkx.Graph): Graph to visualize
        title (str): Plot title
        layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        node_size (int): Size of nodes
        node_color (str): Color of nodes
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Determine layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def visualize_coloring(G, coloring, title="Edge Coloring Visualization", layout='spring', 
                     node_size=300, node_color='skyblue', figsize=(10, 8), 
                     cmap='tab20', save_path=None):
    """
    Visualize a graph with edge coloring.
    
    Args:
        G (networkx.Graph): Graph to visualize
        coloring (dict): Dictionary mapping edges to colors
        title (str): Plot title
        layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        node_size (int): Size of nodes
        node_color (str): Color of nodes
        figsize (tuple): Figure size
        cmap (str): Colormap for edge colors
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Determine layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get edge colors
    edge_colors = [coloring.get(e, coloring.get((e[1], e[0]), 0)) for e in G.edges()]
    
    # Max color for colormap
    max_color = max(edge_colors)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    
    # Draw edges with colors
    nx.draw_networkx_edges(
        G, pos, width=3.0, alpha=0.7,
        edge_color=edge_colors, 
        edge_cmap=plt.cm.get_cmap(cmap, max_color + 1)
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap, max_color + 1), 
                              norm=plt.Normalize(0, max_color))
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=range(max_color + 1))
    cbar.set_label('Color')
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def visualize_coloring_comparison(G, colorings, titles, layout='spring', node_size=300, 
                               node_color='skyblue', figsize=(15, 10), cmap='tab20', 
                               save_path=None):
    """
    Visualize multiple edge colorings of the same graph for comparison.
    
    Args:
        G (networkx.Graph): Graph to visualize
        colorings (list): List of coloring dictionaries
        titles (list): List of titles for each coloring
        layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        node_size (int): Size of nodes
        node_color (str): Color of nodes
        figsize (tuple): Figure size
        cmap (str): Colormap for edge colors
        save_path (str): Path to save the figure
    """
    n_colorings = len(colorings)
    fig, axes = plt.subplots(1, n_colorings, figsize=figsize)
    
    # Handle single subplot case
    if n_colorings == 1:
        axes = [axes]
    
    # Determine layout (same for all subplots)
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Find global max color for consistent colormap
    max_color = max([max(coloring.values()) for coloring in colorings])
    
    # Draw each coloring
    for i, (ax, coloring, title) in enumerate(zip(axes, colorings, titles)):
        # Get edge colors
        edge_colors = [coloring.get(e, coloring.get((e[1], e[0]), 0)) for e in G.edges()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
        
        # Draw edges with colors
        nx.draw_networkx_edges(
            G, pos, width=3.0, alpha=0.7,
            edge_color=edge_colors, 
            edge_cmap=plt.cm.get_cmap(cmap, max_color + 1),
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Add title
        ax.set_title(title)
        ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap, max_color + 1), 
                              norm=plt.Normalize(0, max_color))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, ticks=range(max_color + 1), 
                       orientation='horizontal', pad=0.05)
    cbar.set_label('Color')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()