import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNPathPolicy(nn.Module):
    """
    GNN policy for transcript assembly that incorporates graph structure, 
    current path history, and previously collected paths.
    """
    def __init__(self, node_features, edge_features, hidden_dim=64, num_layers=2, device='cpu'):
        super(GNNPathPolicy, self).__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Node and edge feature encoders
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        if edge_features > 0:
            self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        else:
            self.edge_encoder = None
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if edge_features > 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        # Path encoding components
        self.path_position_encoder = nn.Embedding(20, hidden_dim)  # Position-aware encoding (max 20 nodes)
        self.path_combiner = nn.Linear(hidden_dim, hidden_dim)
        
        # Scoring network
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
    
    def get_node_embeddings(self, state):
        """Process graph to get node embeddings"""
        x = state.x.float().to(self.device)
        edge_index = state.edge_index.to(self.device)
        x = self.node_encoder(x)
        
        edge_attr = None
        if hasattr(state, 'edge_attr') and state.edge_attr is not None and self.edge_encoder is not None:
            edge_attr = state.edge_attr.float().to(self.device)
            edge_attr = self.edge_encoder(edge_attr)
        
        for conv in self.convs:
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
        
        return x
    
    def encode_path(self, path, node_embeddings):
        """Encode a path using position-aware node embeddings"""
        if not path:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        # Limit path length and get embeddings
        path = path[-20:] if len(path) > 20 else path
        path_node_embeddings = node_embeddings[path]
        
        # Add position information
        positions = torch.arange(len(path), device=self.device)
        position_embeddings = self.path_position_encoder(positions)
        combined = path_node_embeddings + position_embeddings
        
        # Weighted average with more weight on recent nodes
        weights = torch.linspace(0.5, 1.0, len(path), device=self.device)
        weighted_sum = torch.sum(weights.unsqueeze(1) * combined, dim=0) / weights.sum()
        
        return self.path_combiner(weighted_sum)
    
    def encode_collected_paths(self, paths, node_embeddings):
        """Encode previously collected paths"""
        if not paths:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        path_embeddings = []
        for path in paths:
            path_embeddings.append(self.encode_path(path, node_embeddings))
        
        return torch.mean(torch.stack(path_embeddings), dim=0)
    
    def forward(self, state, valid_actions, current_node, current_partial_path=None, paths_collected=None):
        """Score valid actions based on graph structure and path history"""
        if not valid_actions:
            return torch.tensor([], device=self.device)
        
        # Process graph and paths
        node_embeddings = self.get_node_embeddings(state)
        
        # Get current path embedding
        path_embedding = self.encode_path(current_partial_path or [], node_embeddings)
        
        # Get collected paths embedding
        collected_paths_embedding = self.encode_collected_paths(paths_collected or [], node_embeddings)
        
        # Score each valid action
        scores = []
        for action in valid_actions:
            # Combine all features for decision making
            combined = torch.cat([
                node_embeddings[current_node],
                node_embeddings[action],
                path_embedding,
                collected_paths_embedding
            ], dim=0)
            
            scores.append(self.score_network(combined))
        
        # Convert to probabilities
        scores = torch.cat(scores) if scores else torch.tensor([], device=self.device)
        probs = F.softmax(scores, dim=0) if len(scores) > 0 else scores
        
        return probs