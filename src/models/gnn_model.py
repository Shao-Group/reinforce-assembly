import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F 


class GNNForRLAgent(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_heads, num_gat_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Normalize input node and edge features
        self.input_norm = nn.BatchNorm1d(num_node_features)
        self.edge_norm = nn.BatchNorm1d(num_edge_features)

        # Define GAT layers. The first layer converts the input node features into a hidden space.
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(num_node_features, hidden_dim, heads=num_heads, edge_dim=num_edge_features)
        )
        # For subsequent layers, the input dimension is (hidden_dim * num_heads)
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=num_edge_features)
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_heads + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Normalize the features
        x = self.input_norm(x)
        edge_attr = self.edge_norm(edge_attr)

        # Pass through the GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            x = F.elu(x)  # non-linear activation

        # Now, x has shape [num_nodes, hidden_dim * num_heads]
        # Get the embeddings for the source and target nodes for every edge.
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]
        h_source = x[source_nodes]    # [num_edges, hidden_dim * num_heads]
        h_target = x[target_nodes]    # [num_edges, hidden_dim * num_heads]

        # Concatenate the source and target embeddings with the edge features.
        edge_input = torch.cat([h_source, h_target, edge_attr], dim=-1)
        # edge_input now has shape: [num_edges, 2*hidden_dim*num_heads + num_edge_features]

        # Pass through the final MLP to produce edge logits.
        edge_logits = self.edge_mlp(edge_input).squeeze(-1)  # [num_edges]

        # The output logits can be used directly in the RL agent for masking and computing softmax.
        return edge_logits