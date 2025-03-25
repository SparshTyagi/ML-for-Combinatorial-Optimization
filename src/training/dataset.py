import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import json

class EdgeColoringDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for edge coloring problems.
    """
    
    def __init__(self, root, graphs=None, colorings=None, transform=None, pre_transform=None, 
                 pre_filter=None, processed_file_prefix='edge_coloring_data', seed=42):
        """
        Initialize the dataset.
        
        Args:
            root (str): Root directory where the dataset should be saved
            graphs (list): List of networkx graphs
            colorings (list): List of edge colorings for each graph
            transform (callable): Transform to be applied on each data object
            pre_transform (callable): Transform to be applied on each data object before saving
            pre_filter (callable): Function that filters unwanted data objects
            processed_file_prefix (str): Prefix for processed data files
            seed (int): Random seed for data splitting
        """
        self.graphs = graphs
        self.colorings = colorings
        self.processed_file_prefix = processed_file_prefix
        self.seed = seed
        
        super(EdgeColoringDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        # If graphs and colorings are provided, process and save them
        if graphs is not None and colorings is not None:
            self.process()
            
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        """List of raw files."""
        return []
    
    @property
    def processed_file_names(self):
        """List of processed files."""
        return [f'{self.processed_file_prefix}.pt']
    
    def process(self):
        """Process the raw data and save it."""
        if self.graphs is None or self.colorings is None:
            raise ValueError("Graphs and colorings must be provided")
            
        data_list = []
        
        # Convert each graph and coloring to a PyTorch Geometric Data object
        for i, (graph, coloring) in enumerate(zip(self.graphs, self.colorings)):
            # Extract node features (placeholder - should be computed from graph)
            node_features = np.zeros((graph.number_of_nodes(), 4))
            for j, node in enumerate(graph.nodes()):
                node_features[j, 0] = graph.degree(node)  # Degree
                node_features[j, 1] = nx.clustering(graph, node)  # Clustering coefficient
                # Add more node features as needed
                
            # Extract edge features (placeholder - should be computed from graph)
            edge_features = np.zeros((graph.number_of_edges(), 4))
            for j, edge in enumerate(graph.edges()):
                u, v = edge
                edge_features[j, 0] = graph.degree(u) + graph.degree(v)  # Sum of endpoint degrees
                # Add more edge features as needed
                
            # Extract edge colors
            edge_colors = np.zeros(graph.number_of_edges())
            for j, edge in enumerate(graph.edges()):
                edge_colors[j] = coloring[edge]
                
            # Convert to PyTorch tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor([[u, v] for u, v in graph.edges()], dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            y = torch.tensor(edge_colors, dtype=torch.long)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                       num_nodes=graph.number_of_nodes())
            
            # Add to list
            data_list.append(data)
            
        # Apply pre-filter and pre-transform if provided
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify_by=None):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            stratify_by (str): Feature to stratify by
            
        Returns:
            tuple: (train_idx, val_idx, test_idx)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n = len(self)
        indices = np.arange(n)
        
        # Extract stratification features if provided
        if stratify_by is not None:
            strat_features = []
            for i in range(n):
                data = self.get(i)
                # Extract relevant feature for stratification
                if stratify_by == 'num_nodes':
                    strat_features.append(data.num_nodes)
                elif stratify_by == 'num_edges':
                    strat_features.append(data.edge_index.shape[1] // 2)
                elif stratify_by == 'max_degree':
                    strat_features.append(int(data.x[:, 0].max().item()))
                else:
                    strat_features.append(0)  # Default
            
            # Convert to categories for stratification
            from sklearn.preprocessing import KBinsDiscretizer
            discretizer = KBinsDiscretizer(n_bins=min(10, n), encode='ordinal', strategy='quantile')
            strat_features = discretizer.fit_transform(np.array(strat_features).reshape(-1, 1)).flatten()
        else:
            strat_features = None
            
        # First split: train+val and test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=self.seed, stratify=strat_features
        )
        
        # Second split: train and val
        if strat_features is not None:
            strat_features_subset = strat_features[train_val_idx]
        else:
            strat_features_subset = None
            
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adjusted, random_state=self.seed, stratify=strat_features_subset
        )
        
        return train_idx, val_idx, test_idx
    
    def save_split(self, train_idx, val_idx, test_idx, filepath):
        """
        Save dataset split to disk.
        
        Args:
            train_idx (list): Training indices
            val_idx (list): Validation indices
            test_idx (list): Test indices
            filepath (str): Path to save the split
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        split_dict = {
            'train': train_idx.tolist() if isinstance(train_idx, np.ndarray) else train_idx,
            'val': val_idx.tolist() if isinstance(val_idx, np.ndarray) else val_idx,
            'test': test_idx.tolist() if isinstance(test_idx, np.ndarray) else test_idx
        }
        with open(filepath, 'w') as f:
            json.dump(split_dict, f)
            
    def load_split(self, filepath):
        """
        Load dataset split from disk.
        
        Args:
            filepath (str): Path to the saved split
            
        Returns:
            tuple: (train_idx, val_idx, test_idx)
        """
        with open(filepath, 'r') as f:
            split_dict = json.load(f)
        return np.array(split_dict['train']), np.array(split_dict['val']), np.array(split_dict['test'])


class TabularEdgeColoringDataset:
    """
    Dataset for edge coloring problems using tabular features.
    This is suitable for traditional ML models like Random Forest.
    """
    
    def __init__(self, graphs=None, colorings=None, edge_features=None, node_features=None):
        """
        Initialize the dataset.
        
        Args:
            graphs (list): List of networkx graphs
            colorings (list): List of edge colorings for each graph
            edge_features (dict): Pre-computed edge features
            node_features (dict): Pre-computed node features
        """
        self.graphs = graphs
        self.colorings = colorings
        self.edge_features = edge_features
        self.node_features = node_features
        
        self.X = None
        self.y = None
        self.feature_names = None
        
    def prepare_features(self, include_node_features=True, include_graph_features=True):
        """
        Prepare features for training.
        
        Args:
            include_node_features (bool): Whether to include node features
            include_graph_features (bool): Whether to include graph-level features
            
        Returns:
            tuple: (X, y, feature_names)
        """
        if self.graphs is None or self.colorings is None:
            raise ValueError("Graphs and colorings must be provided")
            
        all_features = []
        all_labels = []
        feature_names = []
        
        # Process each graph and its coloring
        for graph_idx, (graph, coloring) in enumerate(zip(self.graphs, self.colorings)):
            # Extract features for each edge
            for edge_idx, edge in enumerate(graph.edges()):
                edge_feats = []
                
                # Add edge-specific features
                if self.edge_features is not None:
                    edge_key = (graph_idx, edge)
                    if edge_key in self.edge_features:
                        edge_feats.extend(self.edge_features[edge_key])
                        if len(feature_names) < len(edge_feats):
                            feature_names = [f"edge_feat_{i}" for i in range(len(edge_feats))]
                else:
                    # Compute basic edge features
                    u, v = edge
                    edge_feats.extend([
                        graph.degree(u),
                        graph.degree(v),
                        graph.degree(u) + graph.degree(v)
                    ])
                    if len(feature_names) < len(edge_feats):
                        feature_names = ["degree_u", "degree_v", "sum_degrees"]
                
                # Add node features if requested
                if include_node_features and self.node_features is not None:
                    u, v = edge
                    u_key, v_key = (graph_idx, u), (graph_idx, v)
                    
                    if u_key in self.node_features and v_key in self.node_features:
                        u_feats = self.node_features[u_key]
                        v_feats = self.node_features[v_key]
                        
                        # Combine node features (e.g., concatenate, average)
                        for i, (uf, vf) in enumerate(zip(u_feats, v_feats)):
                            edge_feats.extend([uf, vf, (uf + vf) / 2])
                            if len(feature_names) < len(edge_feats):
                                feature_names.extend([
                                    f"node_feat_{i}_u", f"node_feat_{i}_v", f"node_feat_{i}_avg"
                                ])
                
                # Add graph-level features if requested
                if include_graph_features:
                    # Add basic graph features
                    graph_feats = [
                        graph.number_of_nodes(),
                        graph.number_of_edges(),
                        graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2)  # Density
                    ]
                    edge_feats.extend(graph_feats)
                    
                    if len(feature_names) < len(edge_feats):
                        feature_names.extend(["num_nodes", "num_edges", "density"])
                
                # Add to dataset
                all_features.append(edge_feats)
                all_labels.append(coloring[edge])
        
        # Convert to numpy arrays
        self.X = np.array(all_features)
        self.y = np.array(all_labels)
        self.feature_names = feature_names
        
        return self.X, self.y, self.feature_names
    
    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=False, seed=42):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            stratify (bool): Whether to stratify by labels
            seed (int): Random seed
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.X is None or self.y is None:
            self.prepare_features()
            
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # First split: train+val and test
        stratify_y = self.y if stratify else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_ratio, random_state=seed, stratify=stratify_y
        )
        
        # Second split: train and val
        if stratify:
            stratify_y = y_train_val
        else:
            stratify_y = None
            
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=seed, stratify=stratify_y
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save(self, filepath):
        """
        Save the dataset to disk.
        
        Args:
            filepath (str): Path to save the dataset
        """
        if self.X is None or self.y is None:
            self.prepare_features()
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, X=self.X, y=self.y, feature_names=self.feature_names)
        
    @classmethod
    def load(cls, filepath):
        """
        Load a dataset from disk.
        
        Args:
            filepath (str): Path to the saved dataset
            
        Returns:
            TabularEdgeColoringDataset: Loaded dataset
        """
        data = np.load(filepath, allow_pickle=True)
        dataset = cls()
        dataset.X = data['X']
        dataset.y = data['y']
        dataset.feature_names = data['feature_names']
        return dataset