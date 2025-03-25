import torch
import networkx as nx
import numpy as np
import random
import time
import os
import logging
from src.models.random_forest import EdgeColoringRandomForest
from src.models.gnn import GNNEdgeColoring
from src.coloring.greedy import greedy_edge_coloring
from src.coloring.local_search import local_search_coloring

class HybridEdgeColoring:
    """
    Hybrid approach combining machine learning predictions with traditional optimization.
    """
    
    def __init__(self, base_model_type='gnn', ml_guidance_weight=0.7, local_search_iterations=100,
                 ml_model_params=None, random_seed=42):
        """
        Initialize the hybrid edge coloring model.
        
        Args:
            base_model_type (str): Type of ML model ('random_forest' or 'gnn')
            ml_guidance_weight (float): Weight given to ML predictions (0-1)
            local_search_iterations (int): Number of local search iterations
            ml_model_params (dict): Parameters for the ML model
            random_seed (int): Random seed for reproducibility
        """
        self.base_model_type = base_model_type
        self.ml_guidance_weight = ml_guidance_weight
        self.local_search_iterations = local_search_iterations
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize base ML model
        if base_model_type == 'random_forest':
            self.ml_model = EdgeColoringRandomForest(**(ml_model_params or {}))
        elif base_model_type == 'gnn':
            self.ml_model = GNNEdgeColoring(**(ml_model_params or {}))
        else:
            raise ValueError(f"Unknown base model type: {base_model_type}")
    
    def train(self, data, train_indices, val_indices, **kwargs):
        """
        Train the ML component of the hybrid model.
        
        Args:
            data: Dataset
            train_indices: Indices for training
            val_indices: Indices for validation
            **kwargs: Additional training arguments
            
        Returns:
            self: Trained model
        """
        self.ml_model.train(data, train_indices, val_indices, **kwargs)
        return self
    
    def predict(self, graph, edge_features=None, **kwargs):
        """
        Predict an edge coloring using the hybrid approach.
        
        Args:
            graph (networkx.Graph): Input graph
            edge_features (dict): Pre-computed edge features
            **kwargs: Additional prediction arguments
            
        Returns:
            dict: Edge coloring (mapping edges to colors)
        """
        start_time = time.time()
        
        # Step 1: Get ML model predictions
        ml_predictions = self._get_ml_predictions(graph, edge_features, **kwargs)
        
        # Step 2: Use ML predictions to guide edge ordering for greedy coloring
        guided_coloring = self._guided_greedy_coloring(graph, ml_predictions)
        
        # Step 3: Apply local search to improve the solution
        final_coloring = self._apply_local_search(graph, guided_coloring)
        
        logging.info(f"Hybrid coloring completed in {time.time() - start_time:.2f} seconds.")
        logging.info(f"Number of colors used: {len(set(final_coloring.values()))}")
        
        return final_coloring
    
    def _get_ml_predictions(self, graph, edge_features=None, **kwargs):
        """
        Get color predictions from the ML model.
        
        Args:
            graph (networkx.Graph): Input graph
            edge_features (dict): Pre-computed edge features
            **kwargs: Additional prediction arguments
            
        Returns:
            dict: Edge predictions (can be colors or scores)
        """
        return self.ml_model.predict(graph, **kwargs)
    
    def _guided_greedy_coloring(self, graph, ml_predictions):
        """
        Use ML predictions to guide greedy coloring.
        
        Args:
            graph (networkx.Graph): Input graph
            ml_predictions (dict): ML model predictions
            
        Returns:
            dict: Edge coloring
        """
        # Create a score for each edge based on ML predictions
        edge_scores = {}
        
        for edge in graph.edges():
            # Get the prediction for this edge
            pred = ml_predictions.get(edge, ml_predictions.get((edge[1], edge[0]), 0))
            
            # Convert prediction to a score for edge ordering
            if isinstance(pred, (int, np.integer)):
                # If prediction is a color, use it directly
                edge_scores[edge] = pred
            elif isinstance(pred, dict):
                # If prediction is a dictionary of color probabilities
                edge_scores[edge] = max(pred.items(), key=lambda x: x[1])[0]
            else:
                # Default case
                edge_scores[edge] = pred
        
        # Sort edges based on a combination of ML prediction and graph structure
        # Higher degree edges and edges with more constraints should be colored first
        edge_ordering = []
        for edge in graph.edges():
            # Calculate a priority score (lower is higher priority)
            u, v = edge
            degree_sum = graph.degree(u) + graph.degree(v)
            ml_score = edge_scores.get(edge, 0)
            
            # Combine ML guidance with structural information
            priority = (1 - self.ml_guidance_weight) * (-degree_sum) + self.ml_guidance_weight * ml_score
            edge_ordering.append((edge, priority))
        
        # Sort edges by priority (lower is higher priority)
        sorted_edges = [e for e, _ in sorted(edge_ordering, key=lambda x: x[1])]
        
        # Apply greedy coloring with the sorted edge order
        return greedy_edge_coloring(graph, edge_order=sorted_edges)
    
    def _apply_local_search(self, graph, initial_coloring):
        """
        Apply local search to improve the coloring.
        
        Args:
            graph (networkx.Graph): Input graph
            initial_coloring (dict): Initial edge coloring
            
        Returns:
            dict: Improved edge coloring
        """
        return local_search_coloring(
            graph, initial_coloring, max_iterations=self.local_search_iterations
        )
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the ML component
        ml_filepath = os.path.join(os.path.dirname(filepath), 
                                 f"{os.path.basename(filepath)}_ml_component")
        self.ml_model.save_model(ml_filepath)
        
        # Save hybrid model configuration
        config = {
            'base_model_type': self.base_model_type,
            'ml_guidance_weight': self.ml_guidance_weight,
            'local_search_iterations': self.local_search_iterations,
            'random_seed': self.random_seed,
            'ml_filepath': ml_filepath
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: The loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        # Update configuration
        self.base_model_type = config['base_model_type']
        self.ml_guidance_weight = config['ml_guidance_weight']
        self.local_search_iterations = config['local_search_iterations']
        self.random_seed = config['random_seed']
        
        # Initialize and load the ML component
        if self.base_model_type == 'random_forest':
            self.ml_model = EdgeColoringRandomForest()
        elif self.base_model_type == 'gnn':
            self.ml_model = GNNEdgeColoring()
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
        
        self.ml_model.load_model(config['ml_filepath'])
        
        return self