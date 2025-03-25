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

def plot_scaling_behavior(results, figsize=(12, 8), title="Scaling Behavior by Graph Size",
                         metric='time', save_path=None):
    """
    Visualize how algorithm performance scales with graph size.
    
    Args:
        results (dict): Results dictionary containing performance metrics
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        metric (str): Metric to visualize ('time' by default)
        save_path (str): Path to save the figure
        
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    methods = []
    method_data = {}
    
    for method_name, method_results in results.items():
        if len(method_results[metric]) > 0:  # Only include methods with data
            methods.append(method_name)
            
            # Extract data points
            sizes = [info['num_nodes'] for info in method_results['graph_info']]
            metric_values = method_results[metric]
            
            # Group by size and calculate mean
            df = pd.DataFrame({
                'size': sizes,
                metric: metric_values
            })
            grouped = df.groupby('size')[metric].mean().reset_index()
            
            method_data[method_name] = {
                'sizes': grouped['size'].values,
                'values': grouped[metric].values
            }
    
    # Plot data
    for method in methods:
        ax.plot(
            method_data[method]['sizes'], 
            method_data[method]['values'], 
            marker='o', 
            label=method,
            linewidth=2,
            markersize=6
        )
    
    # Add reference scaling lines if plotting time
    if metric == 'time' and len(method_data) > 0:
        # Find a representative method to base reference lines on
        ref_method = methods[0]
        x = method_data[ref_method]['sizes']
        y = method_data[ref_method]['values']
        
        if len(x) > 1:
            # Get a scaling factor from the data
            scale_factor = y[-1] / 50
            max_x = max(x)
            
            # Plot O(n) reference
            ax.plot(x, [scale_factor * xi for xi in x], 'k--', alpha=0.3, label='O(n)')
            
            # Plot O(n²) reference
            ax.plot(x, [scale_factor * (xi/x[0])**2 for xi in x], 'k:', alpha=0.3, label='O(n²)')
    
    # Set labels and title
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Computation Time (s)' if metric == 'time' else metric.capitalize())
    ax.set_title(title)
    
    # Use log scale for time plots with wide range
    if metric == 'time':
        values = np.concatenate([data['values'] for data in method_data.values()])
        if max(values) / (min(values) + 1e-10) > 100:  # If range is more than 2 orders of magnitude
            ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_solution_quality(results, figsize=(12, 8), title="Solution Quality Comparison",
                         metric='color_count_ratio', by_graph_type=True, save_path=None):
    """
    Visualize the quality of solutions produced by different methods.
    
    Args:
        results (dict): Results dictionary containing performance metrics
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        metric (str): Metric to visualize ('color_count_ratio' by default)
        by_graph_type (bool): Whether to group results by graph type
        save_path (str): Path to save the figure
        
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract and organize data
    df_rows = []
    
    for method_name, method_results in results.items():
        if 'graph_info' in method_results and len(method_results['graph_info']) > 0:
            for i, info in enumerate(method_results['graph_info']):
                if i < len(method_results[metric if metric != 'color_count_ratio' else 'color_count']):
                    value = (method_results[metric][i] if metric != 'color_count_ratio' 
                            else info.get('color_count_ratio', 
                                        method_results['color_count'][i] / info.get('max_degree', 1)))
                    
                    df_rows.append({
                        'Method': method_name,
                        'Graph Type': info['graph_type'],
                        'Value': value,
                        'Num Nodes': info['num_nodes']
                    })
    
    if not df_rows:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    df = pd.DataFrame(df_rows)
    
    # Plot based on grouping preference
    if by_graph_type:
        # Group by graph type and method, show boxplot
        sns.boxplot(x='Graph Type', y='Value', hue='Method', data=df, ax=ax)
        
        # Adjust x-axis labels if they're too long
        if df['Graph Type'].str.len().max() > 15:
            plt.xticks(rotation=45, ha='right')
    else:
        # Just compare methods
        sns.boxplot(x='Method', y='Value', data=df, ax=ax)
    
    # Set labels and title
    y_label = {
        'color_count': 'Number of Colors Used',
        'color_count_ratio': 'Color Count / Maximum Degree',
        'time': 'Computation Time (s)'
    }.get(metric, metric.capitalize())
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if by_graph_type:
        ax.legend(title='Method', loc='upper right')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_performance_comparison(results, metric='color_count_ratio', figsize=(10, 6),
                               title="Algorithm Performance Comparison", save_path=None):
    """
    Compare performance of different algorithms.
    
    Args:
        results (dict): Results dictionary containing performance metrics
        metric (str): Metric to compare ('color_count_ratio' by default)
        figsize (tuple): Figure size
        title (str): Plot title
        save_path (str): Path to save the figure
        
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract metric values for each method
    methods = []
    values = []
    
    for method_name, method_results in results.items():
        # Skip methods with no data
        if metric not in method_results or len(method_results[metric]) == 0:
            continue
            
        methods.append(method_name)
        
        if metric == 'color_count_ratio':
            # For this metric, we need to calculate from graph_info
            ratios = []
            for i, info in enumerate(method_results['graph_info']):
                if 'color_count_ratio' in info:
                    ratios.append(info['color_count_ratio'])
                elif i < len(method_results['color_count']) and 'max_degree' in info:
                    ratios.append(method_results['color_count'][i] / info['max_degree'])
            values.append(ratios)
        else:
            values.append(method_results[metric])
    
    # Create violin plot
    violin_parts = ax.violinplot(values, showmedians=True)
    
    # Add better labels
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    
    # Set labels and title
    y_label = {
        'color_count': 'Number of Colors Used',
        'color_count_ratio': 'Color Count / Maximum Degree',
        'time': 'Computation Time (s)'
    }.get(metric, metric.capitalize())
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax