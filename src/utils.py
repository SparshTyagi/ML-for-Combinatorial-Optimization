# utils.py
import time
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import networkx as nx
import os
import json
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx, to_networkx

def setup_logging(log_file=None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def timer(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def save_graph(graph, filename):
    """Save a NetworkX graph to a GraphML file"""
    nx.write_graphml(graph, filename)

def load_graph(filename):
    """Load a NetworkX graph from a GraphML file"""
    return nx.read_graphml(filename)

def save_results(results, filename):
    """Save results to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(filename):
    """Load results from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)