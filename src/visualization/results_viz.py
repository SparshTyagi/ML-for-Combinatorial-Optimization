import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def plot_training_history(history, title="Training History", figsize=(12, 5), save_path=None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history (dict): Dictionary containing training history
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Plot loss if available
    if 'train_loss' in history and len(history['train_loss']) > 0:
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot accuracy if available
    if 'train_acc' in history and len(history['train_acc']) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        
        if 'val_acc' in history and len(history['val_acc']) > 0:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_model_comparison(results, metric='accuracy', bar_width=0.25, figsize=(12, 8), 
                        title="Model Performance Comparison", save_path=None):
    """
    Compare performance of different models across different graph types.
    
    Args:
        results (dict): Dictionary with structure {graph_type: {model_name: {metric: value}}}
        metric (str): Metric to compare
        bar_width (float): Width of bars in the plot
        figsize (tuple): Figure size
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    graph_types = list(results.keys())
    model_names = list(results[graph_types[0]].keys())
    
    x = np.arange(len(graph_types))
    
    for i, model_name in enumerate(model_names):
        values = [results[graph_type][model_name][metric] for graph_type in graph_types]
        plt.bar(x + i * bar_width, values, bar_width, label=model_name)
    
    plt.xlabel('Graph Type')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(x + bar_width * (len(model_names) - 1) / 2, graph_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_feature_importance(feature_importance, feature_names=None, top_n=None, 
                          figsize=(10, 6), title="Feature Importance", 
                          save_path=None):
    """
    Visualize feature importance from a trained model.
    
    Args:
        feature_importance (array): Array of feature importance values
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        figsize (tuple): Figure size
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    # If feature_importance is a dictionary, convert to arrays
    if isinstance(feature_importance, dict):
        if feature_names is None:
            feature_names = list(feature_importance.keys())
        feature_importance = np.array([feature_importance[name] for name in feature_names])
    
    # Use default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Sort by importance
    sorted_idx = np.argsort(feature_importance)
    
    # Select top N features if specified
    if top_n is not None and top_n < len(feature_names):
        sorted_idx = sorted_idx[-top_n:]
    
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_performance_by_graph_size(results, metric='accuracy', figsize=(10, 6), 
                                 title="Performance vs Graph Size", save_path=None):
    """
    Plot model performance across different graph sizes.
    
    Args:
        results (dict): Dictionary with structure {graph_size: {model_name: {metric: value}}}
        metric (str): Metric to compare
        figsize (tuple): Figure size
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    graph_sizes = sorted(results.keys())
    model_names = list(results[graph_sizes[0]].keys())
    
    for model_name in model_names:
        values = [results[size][model_name][metric] for size in graph_sizes]
        plt.plot(graph_sizes, values, marker='o', label=model_name)
    
    plt.xlabel('Graph Size (Number of Vertices)')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_computation_time(data, figsize=(10, 6), title="Computation Time Comparison", 
                        log_scale=False, save_path=None):
    """
    Visualize computation time of different algorithms across graph sizes.
    
    Args:
        data (dict): Dictionary with structure {algorithm: {graph_size: time}}
        figsize (tuple): Figure size
        title (str): Plot title
        log_scale (bool): Whether to use logarithmic scale for y-axis
        save_path (str): Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    algorithms = list(data.keys())
    all_sizes = set()
    for alg in algorithms:
        all_sizes.update(data[alg].keys())
    graph_sizes = sorted(all_sizes)
    
    for alg in algorithms:
        sizes = sorted(data[alg].keys())
        times = [data[alg][size] for size in sizes]
        plt.plot(sizes, times, marker='o', label=alg)
    
    plt.xlabel('Graph Size (Number of Vertices)')
    plt.ylabel('Computation Time (seconds)')
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()