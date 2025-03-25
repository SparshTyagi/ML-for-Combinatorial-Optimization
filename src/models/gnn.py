import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data
import os
import logging

class EdgeColoringGNN(nn.Module):
    """
    Graph Neural Network model for edge coloring predictions.
    """
    
    def __init__(self, node_features=4, edge_features=4, hidden_channels=64, num_layers=3, 
                 gnn_type='gcn', dropout=0.2, max_colors=10):
        """
        Initialize the GNN model.
        
        Args:
            node_features (int): Number of node features
            edge_features (int): Number of edge features
            hidden_channels (int): Size of hidden layers
            num_layers (int): Number of GNN layers
            gnn_type (str): Type of GNN ('gcn', 'gat', or 'graph')
            dropout (float): Dropout probability
            max_colors (int): Maximum number of colors (output dimension)
        """
        super(EdgeColoringGNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.max_colors = max_colors
        
        # Node encoder
        self.node_encoder = nn.Linear(node_features, hidden_channels)
        
        # Edge encoder
        self.edge_encoder = nn.Linear(edge_features, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_channels
            
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_channels, hidden_channels))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(in_channels, hidden_channels))
            elif gnn_type == 'graph':
                self.convs.append(GraphConv(in_channels, hidden_channels))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, max_colors)
        )
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data (torch_geometric.data.Data): Input graph data
        
        Returns:
            torch.Tensor: Predicted scores for each color
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Node feature encoding
        x = self.node_encoder(x)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Edge prediction
        # Get the node features for each edge
        edge_src, edge_dst = edge_index
        edge_features = torch.cat([x[edge_src], x[edge_dst], edge_attr], dim=1)
        
        # Predict edge colors
        return self.edge_predictor(edge_features)

class GNNEdgeColoring:
    """
    Wrapper class for edge coloring using GNN models.
    """
    
    def __init__(self, node_features=4, edge_features=4, hidden_channels=64, num_layers=3, 
                 gnn_type='gcn', dropout=0.2, max_colors=10, device=None):
        """
        Initialize the GNN edge coloring model.
        
        Args:
            node_features (int): Number of node features
            edge_features (int): Number of edge features
            hidden_channels (int): Size of hidden layers
            num_layers (int): Number of GNN layers
            gnn_type (str): Type of GNN ('gcn', 'gat', or 'graph')
            dropout (float): Dropout probability
            max_colors (int): Maximum number of colors (output dimension)
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.params = {
            'node_features': node_features,
            'edge_features': edge_features,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'gnn_type': gnn_type,
            'dropout': dropout,
            'max_colors': max_colors
        }
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = EdgeColoringGNN(**self.params).to(self.device)
    
    def train(self, data, train_indices, val_indices, batch_size=32, epochs=100, 
              lr=0.001, weight_decay=5e-4, patience=20, task='edge_classification'):
        """
        Train the GNN model.
        
        Args:
            data: List of PyTorch Geometric Data objects
            train_indices: Indices for training
            val_indices: Indices for validation
            batch_size (int): Batch size
            epochs (int): Number of training epochs
            lr (float): Learning rate
            weight_decay (float): Weight decay for L2 regularization
            patience (int): Patience for early stopping
            task (str): Task type ('edge_classification' or 'graph_regression')
            
        Returns:
            self: Trained model
        """
        # Implement training code here, using src.training.train_evaluate
        from src.training.train_evaluate import train_model
        
        # Train the model
        self.model, self.history = train_model(
            self.model, 
            [data[i] for i in train_indices], 
            [data[i] for i in val_indices], 
            model_type='gnn',
            task=task,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=self.device,
            early_stopping=patience
        )
        
        return self
    
    def predict(self, graph_data):
        """
        Predict edge colors for a graph.
        
        Args:
            graph_data: PyTorch Geometric Data object
        
        Returns:
            dict: Edge coloring (mapping edges to colors)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            graph_data = graph_data.to(self.device)
            
            # Make predictions
            outputs = self.model(graph_data)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            # Create coloring dictionary
            coloring = {}
            for i, (src, dst) in enumerate(graph_data.edge_index.t().cpu().numpy()):
                coloring[(int(src), int(dst))] = int(predictions[i])
            
            return coloring
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: The loaded model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model with the same parameters
        self.params = checkpoint['params']
        self.model = EdgeColoringGNN(**self.params).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self