from src.data_loader import *
from src.models.gnn_model import *

#TODO

# num_node_features = data_obj.x.size(1)
# num_edge_features = data_obj.edge_attr.size(1)

# hidden_dim = 64   
# num_heads = 4     
# num_gat_layers = 3  

# model = GNNForRLAgent(num_node_features, num_edge_features, hidden_dim, num_heads, num_gat_layers)

# model.eval()

def debug_forward(model, data):
    print("=== Debug Forward Pass ===")
    print("Input node features shape:", data.x.shape)
    print("Input edge features shape:", data.edge_attr.shape)
    print("Input edge index shape:", data.edge_index.shape)
    print("-" * 50)
    
    # Normalize features
    x = model.input_norm(data.x)
    edge_attr = model.edge_norm(data.edge_attr)
    print("After normalization:")
    print("Node features shape:", x.shape)
    print("Edge features shape:", edge_attr.shape)
    print("-" * 50)
    
    # Pass through each GAT layer with activation
    for i, gat_layer in enumerate(model.gat_layers):
        x = gat_layer(x, data.edge_index, edge_attr)
        x = F.elu(x)
        print(f"After GAT layer {i + 1}:")
        print("Node features shape:", x.shape)
        print("-" * 50)
    
    # Extract node embeddings for each edge
    source_nodes = data.edge_index[0]
    target_nodes = data.edge_index[1]
    h_source = x[source_nodes]
    h_target = x[target_nodes]
    print("Source node embeddings shape:", h_source.shape)
    print("Target node embeddings shape:", h_target.shape)
    print("-" * 50)
    
    # Concatenate source and target embeddings with edge features
    edge_input = torch.cat([h_source, h_target, edge_attr], dim=-1)
    print("Concatenated edge input shape:", edge_input.shape)
    print("-" * 50)
    
    # Compute edge logits via the final MLP
    edge_logits = model.edge_mlp(edge_input).squeeze(-1)
    print("Final edge logits shape:", edge_logits.shape)
    print("Edge logits:", edge_logits)
    print("=" * 50)
    
    return edge_logits


#TODO

# Run the debug-forward pass on your already-loaded data object
# edge_logits = debug_forward(model, data_obj)