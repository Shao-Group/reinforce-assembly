import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import argparse

def evaluate_benchmark(data_list):
    all_true_labels = []
    all_predicted_scores = []

    for data in data_list:
        true_labels = data.y.numpy()
        abundance_scores = data.abundance.numpy()

        all_true_labels.extend(true_labels)
        all_predicted_scores.extend(abundance_scores)

    #print(all_true_labels)
    #print(all_predicted_scores)
    positive_cases = sum(1 for label in all_true_labels if label == 1)
    negative_cases = sum(1 for label in all_true_labels if label == 0)
    benchmark_precision = positive_cases / len(all_true_labels) if len(all_true_labels) > 0 else 0.0
    print(f"Positive Cases: {positive_cases}, Negative Cases: {negative_cases}, Benchmark_precision: {benchmark_precision:.4f}")

    precision, recall, _ = precision_recall_curve(all_true_labels, all_predicted_scores)
    fpr, tpr, _ = roc_curve(all_true_labels, all_predicted_scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    return precision, recall, fpr, tpr, roc_auc, pr_auc

def plot_auc_benchmark(train_labels, train_preds, val_labels, val_preds, benchmark_train_data, benchmark_test_data, runID):
    # Calculate ROC and PR for training
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_preds)
    precision_train, recall_train, _ = precision_recall_curve(train_labels, train_preds)

    # Calculate ROC and PR for validation
    fpr_val, tpr_val, _ = roc_curve(val_labels, val_preds)
    precision_val, recall_val, _ = precision_recall_curve(val_labels, val_preds)

    # Unpack benchmark data
    precision_benchmark_train, recall_benchmark_train, fpr_benchmark_train, tpr_benchmark_train, roc_auc_benchmark_train, pr_auc_benchmark_train = benchmark_train_data
    precision_benchmark_test, recall_benchmark_test, fpr_benchmark_test, tpr_benchmark_test, roc_auc_benchmark_test, pr_auc_benchmark_test = benchmark_test_data

    # Plot ROC curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, label='Train ROC (Model)', color='blue')
    plt.plot(fpr_val, tpr_val, label='Val ROC (Model)', color='orange')
    plt.plot(fpr_benchmark_train, tpr_benchmark_train, label='Train ROC (Benchmark)', linestyle='--', color='blue')
    plt.plot(fpr_benchmark_test, tpr_benchmark_test, label='Val ROC (Benchmark)', linestyle='--', color='orange')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot PR curves
    plt.subplot(1, 2, 2)
    plt.plot(recall_train, precision_train, label='Train PR (Model)', color='blue')
    plt.plot(recall_val, precision_val, label='Val PR (Model)', color='orange')
    plt.plot(recall_benchmark_train, precision_benchmark_train, label='Train PR (Benchmark)', linestyle='--', color='blue')
    plt.plot(recall_benchmark_test, precision_benchmark_test, label='Val PR (Benchmark)', linestyle='--', color='orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'/storage/work/qzs23/PathEm/plot/{runID}-auc_benchmark.png')


def plot_all_losses(all_train_losses, all_val_losses, prev_epochs, runID):
    """
    Plots the training and validation losses over epochs, distinguishing between
    previous and current training sessions.

    Args:
    - all_train_losses (list or np.array): Combined list of all training losses (previous + current).
    - all_val_losses (list or np.array): Combined list of all validation losses (previous + current).
    - prev_epochs (int): Number of epochs in the previous training session to mark the boundary.
    """

    # Create epoch numbers
    epochs = np.arange(1, len(all_train_losses) + 1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, all_train_losses, label='Training Loss', color='b')
    plt.plot(epochs, all_val_losses, label='Validation Loss', color='g')

    # Add a vertical line to separate previous and new training
    if prev_epochs > 0:
        plt.axvline(x=prev_epochs, color='r', linestyle='--', label='New Training Start')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Across All Sessions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/storage/work/qzs23/PathEm/plot/{runID}-loss_curve.png')
    plt.show()
    plt.close()

def load_data(input_dir, prefix, samples=None):
    # Define file paths
    node_file = os.path.join(input_dir, f'{prefix}.node.csv')
    edge_file = os.path.join(input_dir, f'{prefix}.edge.csv')
    path_file = os.path.join(input_dir, f'{prefix}.path.label.csv')
    phasing_file = os.path.join(input_dir, f'{prefix}.phasing.csv')
    
    # Define dtypes
    node_dtypes = {"chr": str, "graph_id": str, "node_id": int, "start_pos": np.uint64, "end_pos": np.uint64, 
                   "weight": float, "length": int, "maxcov": float, "stddev": float, "indel_sum_cov": float, 
                   "indel_ratio": float, "left_indel": int, "right_indel": int, "sample": str}
    edge_dtypes = {"chr": str, "graph_id": str, "source": int, "target": int, "start_pos": np.uint64, 
                   "end_pos": np.uint64, "weight": float, "length": int, "sample": str}
    path_dtypes = {"chr": str, "graph_id": str, "path_id": str, "node_sequence": str, "splice_source": str, 
                   "splice_target": str, "abundance": float, "label": int, "sample": str}
    phasing_dtypes = {"chr": str, "graph_id": str, "path_id": str, "node_sequence": str, "count": int, "sample": str}
    
    # Read CSV files with headers
    node_df = pd.read_csv(node_file, dtype=node_dtypes)
    edge_df = pd.read_csv(edge_file, dtype=edge_dtypes)
    path_df = pd.read_csv(path_file, dtype=path_dtypes)
    phasing_df = pd.read_csv(phasing_file, dtype=phasing_dtypes)
    
    # Filter by samples if specified
    if samples:
        node_df = node_df[node_df['sample'].isin(samples)]
        edge_df = edge_df[edge_df['sample'].isin(samples)]
        path_df = path_df[path_df['sample'].isin(samples)]
        phasing_df = phasing_df[phasing_df['sample'].isin(samples)]
    
    return node_df, edge_df, path_df, phasing_df

def process_graph_to_data(node_df, edge_df, path_df, phasing_df, graph_id):
    # Filter data for the specific graph
    nodes = node_df[node_df['graph_id'] == graph_id]
    edges = edge_df[edge_df['graph_id'] == graph_id]
    paths = path_df[path_df['graph_id'] == graph_id]
    phasing = phasing_df[phasing_df['graph_id'] == graph_id]

    # Test for small graphs
    if(len(nodes) < 5 or len(nodes) > 500):
        return None

    # Check for duplicate graph_id
    #duplicate_node_zero = nodes[nodes['node_id'] == 0]
    #if len(duplicate_node_zero) > 1:
        #print(f"Warning: Duplicate graph_id '{graph_id}' detected!")
        #return None

    # Create node and edge features
    #node_features = torch.tensor(nodes[['extPathSupport', 'weight', 'length', 'maxcov', 'stddev', 'indel_sum_cov','indel_ratio','left_indel','right_indel']].values, dtype=torch.float)
    node_features = torch.tensor(nodes[['weight', 'length', 'maxcov', 'stddev']].values, dtype=torch.float)
    edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)
    #edge_features = torch.tensor(edges[['extPathSupport', 'weight', 'length']].values, dtype=torch.float)
    edge_features = torch.tensor(edges[['weight', 'length']].values, dtype=torch.float)

    # Process paths and phasing paths
    path_sequences = [list(map(int, seq.split(','))) for seq in paths['node_sequence']]
    path_labels = paths['label'].tolist()
    path_abundances = paths['abundance'].tolist()
    splice_sources = [list(map(int, seq.split(','))) for seq in paths['splice_source']]
    splice_targets = [list(map(int, seq.split(','))) for seq in paths['splice_target']]

    if not path_sequences:
        return None

    # Filter based on splice junctions
    #filtered_paths = []
    #for path, label, abundance, source, target in zip(path_sequences, path_labels, path_abundances, splice_sources, splice_targets):
        #if len(path) >= 3:
            #filtered_paths.append((path, label, abundance, source, target))

    #if not filtered_paths:
        #return None
    
    #path_sequences, path_labels, path_abundances, splice_sources, splice_targets = zip(*filtered_paths)

    # Check if all labels are 0
    if all(label == 0 for label in path_labels):
        return None

    # Process fragments and filter those related to paths
    phasing_sequences = [list(map(int, seq.split(','))) for seq in phasing['node_sequence']]
    fragment_counts = phasing['count'].tolist()
    
    if not phasing_sequences:
        return None
    
    # if fragment is related to any path
    #def is_fragment_related_to_paths(fragment, paths):
        #fragment_set = set(fragment)
        #return any(fragment_set.intersection(set(path)) for path in paths)

    # Filter fragments based on relation to paths and minimum length
    #filtered_frags = [(frag, count) 
                      #for frag, count in zip(phasing_sequences, fragment_counts) 
                      #if len(frag) >= 3 and is_fragment_related_to_paths(frag, path_sequences)]

    #if not filtered_frags:
        #return None

    #phasing_sequences, fragment_counts = zip(*filtered_frags)

    # Convert to PyTorch Geometric Data object
    data = Data(x=node_features, 
                edge_index=edge_index, 
                edge_attr=edge_features)
    data.candidate_paths = path_sequences
    data.fragments = phasing_sequences
    data.fragment_counts = torch.tensor(fragment_counts, dtype=torch.float)
    data.y = torch.tensor(path_labels, dtype=torch.float)
    data.abundance = torch.tensor(path_abundances, dtype=torch.float)
    data.splice_sources = splice_sources
    data.splice_targets = splice_targets

    return data

# Set the environment variable for CuBLAS deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def print_batch_contents(batch):
    print(batch)

    #print("Node offset:")
    #print(batch.ptr)

    #print("Node Features (x):")
    #print(batch.x.shape)

    #print("\nEdge Index (edge_index):")
    #print(batch.edge_index)

    #print("\nEdge Attributes (edge_attr):")
    #print(batch.edge_attr.shape if batch.edge_attr is not None else "None")

    #print("\nBatch Indices (batch):")
    #print(batch.batch if batch.batch is not None else "None")

    #print("\nCandidate Paths (custom attribute):")
    #print(len(batch.candidate_paths))
    #print(batch.candidate_paths)
    #print(batch.candidate_paths.shape)

    #print("\nCandidate Path Sources (custom attribute):")
    #print(len(batch.candidate_paths))
    #print(batch.splice_sources)
    #print(batch.splice_sources.shape)

    #print("\nCandidate Path Targets (custom attribute):")
    #print(len(batch.candidate_paths))
    #print(batch.splice_targets)
    #print(batch.splice_targets.shape)

    #print("\nFragments (custom attribute):")
    #print(len(batch.fragments))
    #print(batch.fragments)
    #print(batch.fragments.shape)

    #print("\nFragment Counts (custom attribute):")
    #print(batch.fragment_counts.shape)
    #print(batch.fragment_counts)

    #print("\nTarget Labels (y):")
    #print(batch.y.shape)
    #print(batch.y)

def batch_path_data(batch):
    # The input 'batch' is a PyTorch Geometric Batch object
    num_nodes = batch.num_nodes
    num_graphs = batch.num_graphs

    # Prepare lists to gather all paths and fragments
    all_paths = []
    all_path_sources = []
    all_path_targets = []
    all_fragments = []

    #print(f"Original batch.candidate_paths = {batch.candidate_paths}")
    #print(f"Original batch.fragments = {batch.fragments}")
    #print(f"Graph offsets = {batch.ptr}")

    # Iterate through each graph in the batch
    for i in range(num_graphs):
        # Node offset for the current graph
        node_offset = batch.ptr[i]

        # Candidate paths and fragments for the current graph
        candidate_paths = batch.candidate_paths[i]
        splice_sources = batch.splice_sources[i]
        splice_targets = batch.splice_targets[i]
        fragments = batch.fragments[i]

        # Append adjusted paths to the all_paths list
        for path in candidate_paths:
            adjusted_path = [node + node_offset for node in path]
            all_paths.append(adjusted_path)
        
        # Append adjusted paths to the all_paths list
        for path_source in splice_sources:
            adjusted_path_source = [node + node_offset for node in path_source]
            all_path_sources.append(adjusted_path_source)
        
        # Append adjusted paths to the all_paths list
        for path_target in splice_targets:
            adjusted_path_target = [node + node_offset for node in path_target]
            all_path_targets.append(adjusted_path_target)

        # Append adjusted fragments to the all_fragments list
        for frag in fragments:
            adjusted_frag = [node + node_offset for node in frag]
            all_fragments.append(adjusted_frag)

    # Determine total number of paths and fragments
    total_num_paths = len(all_paths)
    total_num_fragments = len(all_fragments)
    #print(f"Adjusted path = {all_paths}")
    #print(f"Adjusted fragments = {all_fragments}")

    # Create binary tensors
    batched_candidate_paths = torch.zeros((num_nodes, total_num_paths), dtype=torch.float)
    for path_idx, path in enumerate(all_paths):
        for node in path:
            batched_candidate_paths[node, path_idx] = 1.0
    
    batched_candidate_path_sources = torch.zeros((num_nodes, total_num_paths), dtype=torch.float)
    for path_idx, path in enumerate(all_path_sources):
        for node in path:
            batched_candidate_path_sources[node, path_idx] = 1.0
    
    batched_candidate_path_targets = torch.zeros((num_nodes, total_num_paths), dtype=torch.float)
    for path_idx, path in enumerate(all_path_targets):
        for node in path:
            batched_candidate_path_targets[node, path_idx] = 1.0

    batched_fragments = torch.zeros((num_nodes, total_num_fragments), dtype=torch.float)
    for frag_idx, frag in enumerate(all_fragments):
        for node in frag:
            batched_fragments[node, frag_idx] = 1.0

    # Assign these tensors to the batch object
    batch.candidate_paths = batched_candidate_paths
    batch.splice_sources = batched_candidate_path_sources
    batch.splice_targets = batched_candidate_path_targets
    batch.fragments = batched_fragments

    return batch

class GPUEfficientPathScoringModel(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_heads, hidden_dim_LSTM, num_gat_layers):
        super(GPUEfficientPathScoringModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_dim_LSTM = hidden_dim_LSTM
        self.num_heads = num_heads
        
        self.input_norm = nn.BatchNorm1d(num_node_features)
        self.edge_norm = nn.BatchNorm1d(num_edge_features)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(num_node_features, hidden_dim, heads=num_heads, edge_dim=num_edge_features))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=num_edge_features))

        # LSTM layer for edge-based paths
        #self.path_lstm = nn.LSTM(hidden_dim * num_heads * 2 + num_edge_features, hidden_dim,
        self.node_seq_lstm = nn.LSTM(hidden_dim * num_heads, hidden_dim_LSTM, batch_first=True, num_layers=1)
        self.splice_lstm = nn.LSTM(hidden_dim * num_heads * 2, hidden_dim_LSTM, batch_first=True)
        self.fragment_lstm = nn.LSTM(hidden_dim * num_heads, hidden_dim, batch_first=True, num_layers=1)

        # Final scoring layers
        self.fc1 = nn.Linear(hidden_dim_LSTM * 4+1, hidden_dim_LSTM)
        self.fc2 = nn.Linear(hidden_dim_LSTM, 1)

    def forward(self, x, edge_index, edge_attr, candidate_paths, candidate_paths_source, candidate_paths_sink, path_abundance, fragments, fragment_counts):
        # Apply input normalization
        x = self.input_norm(x)
        edge_attr = self.edge_norm(edge_attr)

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            if i == len(self.gat_layers) - 1:  # Last layer
                x, attention = gat_layer(x, edge_index, edge_attr, return_attention_weights=True)
            else:
                x = gat_layer(x, edge_index, edge_attr)
            x = F.elu(x)

        # Extract attention weights from the last layer
        edge_index, attention_weights = attention

        # Create edge mapping
        edge_mapping = self.create_edge_mapping(edge_index, x.shape[0])

        # Process edge-based paths
        node_seq_embeddings = self.process_node_sequences(x, candidate_paths, self.node_seq_lstm)
        splice_embeddings = self.process_splice_sequences(x, edge_attr, edge_mapping, attention_weights, candidate_paths_source, candidate_paths_sink, self.splice_lstm)
        path_embeddings = torch.cat([node_seq_embeddings, splice_embeddings, path_abundance.unsqueeze(-1)], dim=-1)

        # Process fragments
        fragment_embeddings = self.process_node_sequences(x, fragments, self.fragment_lstm)

        positive_comp_matrix, negative_comp_matrix = self.calculate_compatibility_separate(candidate_paths, fragments)

        # Apply compatibility
        path_fragment_pos = torch.matmul(positive_comp_matrix, fragment_embeddings)
        path_fragment_neg = torch.matmul(negative_comp_matrix, fragment_embeddings)

        # Combine path and processed fragment information
        combined = torch.cat([path_embeddings, path_fragment_pos, path_fragment_neg], dim=-1)

        # Final scoring
        score = self.fc1(combined)
        score = F.relu(score)
        score = self.fc2(score).squeeze(-1)
        probability = torch.sigmoid(score)

        return probability
    
    def process_node_sequences(self, x, mask, lstm):
        # x shape: (#nodes_in_batch, gnn_output_dim)
        # mask shape: (#nodes_in_batch, num_sequences)
        
        device = x.device
        num_nodes, num_sequences = mask.shape
        gnn_output_dim = x.shape[1]

        # Get sequence lengths
        seq_lengths = mask.sum(dim=0).long()
        max_length = seq_lengths.max().item()

        # Create a tensor to hold the sequence embeddings
        seq_embeddings = torch.zeros(num_sequences, max_length, gnn_output_dim, device=device)

        # Use boolean indexing for efficient assignment
        valid_mask = mask.bool()
        
        # Create index tensors
        node_indices = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, num_sequences)
        seq_indices = torch.arange(num_sequences, device=device).unsqueeze(0).expand(num_nodes, -1)
        
        # Compute positions within sequences
        seq_positions = torch.cumsum(valid_mask, dim=0) - 1
        
        # Perform the assignment
        seq_embeddings[seq_indices[valid_mask], seq_positions[valid_mask]] = x[node_indices[valid_mask]]

        # Sort sequences by length for more efficient packing
        seq_lengths, sort_indices = seq_lengths.sort(descending=True)
        seq_embeddings = seq_embeddings[sort_indices]

        # Pack the sequences
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            seq_embeddings, seq_lengths.cpu(), batch_first=True
        )

        # Process with LSTM
        output, (h_n, _) = lstm(packed_seq)

        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get the last actual output for each sequence
        last_output = output[torch.arange(num_sequences, device=device), seq_lengths - 1]

        # Restore original order
        _, unsort_indices = sort_indices.sort()
        last_output = last_output[unsort_indices]

        return last_output
    
    def create_edge_mapping(self, edge_index, num_nodes):
        device = edge_index.device
        num_edges = edge_index.shape[1]
        
        # Create a unique identifier for each edge
        edge_identifier = edge_index[0] * num_nodes + edge_index[1]
        
        # Create a mapping from edge identifier to edge index
        edge_mapping = torch.zeros(edge_identifier.max() + 1, dtype=torch.long, device=device)
        edge_mapping[edge_identifier] = torch.arange(num_edges, device=device)
        
        return edge_mapping
    
    def process_splice_sequences(self, node_embeddings, edge_attr, edge_mapping, attention_weights, source_mask, sink_mask, lstm):
        device = node_embeddings.device
        num_nodes, num_sequences = source_mask.shape
        gnn_output_dim = node_embeddings.shape[1]
        edge_attr_dim = edge_attr.shape[1]

        # Get sequence lengths
        seq_lengths = source_mask.sum(dim=0).long()
        max_length = seq_lengths.max().item()

        # Create a tensor to hold the sequence embeddings
        seq_embeddings = torch.zeros(num_sequences, max_length, gnn_output_dim * 2, device=device)

        # Use boolean indexing for efficient assignment
        valid_mask = source_mask.bool()
        #print(valid_mask.device)
        
        # Create index tensors
        node_indices = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, num_sequences)
        seq_indices = torch.arange(num_sequences, device=device).unsqueeze(0).expand(num_nodes, -1)
        
        # Compute positions within sequences
        seq_positions = torch.cumsum(valid_mask, dim=0) - 1
        
        # Compute edge indices
        edge_identifier = node_indices[source_mask.bool()] * node_embeddings.shape[0] + node_indices[sink_mask.bool()]
        edge_indices = edge_mapping[edge_identifier]

        # Get node embeddings for source and sink nodes
        source_embeddings = node_embeddings[node_indices[source_mask.bool()]]
        sink_embeddings = node_embeddings[node_indices[sink_mask.bool()]]
        #agg_embeddings = source_embeddings + sink_embeddings

        # Get attention weights for the edges (shape: [num_edges, num_heads])
        edge_attention = attention_weights[edge_indices]

        # Compute attention-weighted edge embeddings
        edge_embeddings = torch.cat([
            (source_embeddings.view(-1, self.num_heads, self.hidden_dim) * edge_attention.unsqueeze(-1)).view(-1, gnn_output_dim),
            (sink_embeddings.view(-1, self.num_heads, self.hidden_dim) * edge_attention.unsqueeze(-1)).view(-1, gnn_output_dim)
        ], dim=-1)

        # Assign edge embeddings and attributes to sequence embeddings
        seq_embeddings[seq_indices[valid_mask], seq_positions[valid_mask], :2*gnn_output_dim] = edge_embeddings

        # Sort sequences by length for more efficient packing
        seq_lengths, sort_indices = seq_lengths.sort(descending=True)
        seq_embeddings = seq_embeddings[sort_indices]

        # Pack the sequences
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            seq_embeddings, seq_lengths.cpu(), batch_first=True
        )

        # Process with LSTM
        output, (h_n, _) = lstm(packed_seq)

        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get the last actual output for each sequence
        last_output = output[torch.arange(num_sequences, device=device), seq_lengths - 1]

        # Restore original order
        _, unsort_indices = sort_indices.sort()
        last_output = last_output[unsort_indices]

        return last_output
    
    def calculate_compatibility(self, paths, fragments):
        # paths: (num_nodes, num_paths)
        # fragments: (num_nodes, num_fragments)
        
        num_nodes, num_paths = paths.shape
        num_fragments = fragments.shape[1]
        
        # Step 1: Compute relevance (any shared nodes)
        relevance = torch.mm(paths.t(), fragments).bool()  # (num_paths, num_fragments)

        # Step 2: Find start and end indices for each fragment
        frag_starts = fragments.argmax(dim=0)
        frag_ends = num_nodes - 1 - torch.flip(fragments, [0]).argmax(dim=0)

        # Initialize the result tensor
        result = torch.zeros((num_paths, num_fragments), device=paths.device)

        # Process fragments in batches
        batch_size = 1024  # Secondary batch size to reduce peak GPU memory
        for i in range(0, num_fragments, batch_size):
            batch_end = min(i + batch_size, num_fragments)
            
            # Create masks for the current batch of fragments
            range_tensor = torch.arange(num_nodes, device=paths.device).unsqueeze(1)
            fragment_masks = (range_tensor >= frag_starts[i:batch_end].unsqueeze(0)) & (range_tensor <= frag_ends[i:batch_end].unsqueeze(0))

            # Check for compatibility (subpath condition)
            for j in range(num_paths):
                path = paths[:, j].unsqueeze(1)
                matches = (path == fragments[:, i:batch_end]) | ~fragment_masks
                compatibility = matches.all(dim=0)
                
                # Update the result tensor
                result[j, i:batch_end] = torch.where(
                    relevance[j, i:batch_end] == 1,
                    torch.where(compatibility == 1, torch.tensor(1.0, device=paths.device), torch.tensor(-1.0, device=paths.device)),
                    torch.tensor(0.0, device=paths.device)
                )

        return result
    
    def calculate_compatibility_separate(self, paths, fragments):
        # paths: (num_nodes, num_paths)
        # fragments: (num_nodes, num_fragments)
        num_nodes, num_paths = paths.shape
        num_fragments = fragments.shape[1]
        
        # Step 1: Compute relevance (any shared nodes)
        relevance = torch.mm(paths.t(), fragments).bool()  # (num_paths, num_fragments)
        
        # Step 2: Find start and end indices for each fragment
        frag_starts = fragments.argmax(dim=0)
        frag_ends = num_nodes - 1 - torch.flip(fragments, [0]).argmax(dim=0)
        
        # Initialize separate tensors for positive and negative supports
        positive_supports = torch.zeros((num_paths, num_fragments), device=paths.device)
        negative_supports = torch.zeros((num_paths, num_fragments), device=paths.device)
        
        # Process fragments in batches
        batch_size = 1024  # Secondary batch size to reduce peak GPU memory
        for i in range(0, num_fragments, batch_size):
            batch_end = min(i + batch_size, num_fragments)
            
            # Create masks for the current batch of fragments
            range_tensor = torch.arange(num_nodes, device=paths.device).unsqueeze(1)
            fragment_masks = (range_tensor >= frag_starts[i:batch_end].unsqueeze(0)) & \
                            (range_tensor <= frag_ends[i:batch_end].unsqueeze(0))
            
            # Check for compatibility (subpath condition)
            for j in range(num_paths):
                path = paths[:, j].unsqueeze(1)
                matches = (path == fragments[:, i:batch_end]) | ~fragment_masks
                compatibility = matches.all(dim=0)
                
                # Update the positive and negative support matrices
                is_relevant = relevance[j, i:batch_end]
                is_compatible = compatibility
                
                # Set positive supports (relevant AND compatible)
                positive_supports[j, i:batch_end] = torch.where(
                    is_relevant & is_compatible,
                    torch.tensor(1.0, device=paths.device),
                    torch.tensor(0.0, device=paths.device)
                )
                
                # Set negative supports (relevant AND NOT compatible)
                negative_supports[j, i:batch_end] = torch.where(
                    is_relevant & ~is_compatible,
                    torch.tensor(1.0, device=paths.device),
                    torch.tensor(0.0, device=paths.device)
                )
        
        return positive_supports, negative_supports
    
def train_and_evaluate(model, train_loader, test_loader, args):

    print(f"Device: {args.device}")
    model = model.to(args.device)

    # Load checkpoint if specified
    if args.load_checkpoint and args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        loss_history = checkpoint['loss_history']
        print(f"Resuming training from epoch {start_epoch + 1}")
    elif args.load_pretrain and args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = -1
        best_loss = float('inf')
        loss_history = {'train_losses': [], 'val_losses': []}
    else:
        start_epoch = -1
        best_loss = float('inf')
        loss_history = {'train_losses': [], 'val_losses': []}

    model = model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6, verbose=True)


    criterion = torch.nn.BCEWithLogitsLoss()

    best_auroc = 0
    patience = 30
    epochs_no_improve = 0
    train_losses = loss_history['train_losses']
    val_losses = loss_history['val_losses']

    # To store predictions and labels of the best model
    best_train_preds = []
    best_train_labels = []
    best_val_preds = []
    best_val_labels = []

    # Pre-batch validation data
    
    #batched_test_data = []
    #for batch in test_loader:
        #batch = batch.to(args.device)
        #batched_data = batch_path_data(batch) # Move batch to GPU
        #batched_test_data.append(batched_data)


    # Training loop
    print(f"runID {args.runID}")
    for epoch in range(start_epoch + 1, args.num_epochs):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"===== {current_date}")

        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            batch = batch_path_data(batch)
            batch = batch.to(args.device)
            #print("\n===Train Batches: ")
            #print_batch_contents(batch)

            optimizer.zero_grad()
            # Forward pass
            #output = model(batch.x, batch.edge_index, batch.edge_attr, batch.candidate_paths, batch.fragments, batch.fragment_counts, batch.batch)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.candidate_paths, batch.splice_sources, batch.splice_targets, batch.abundance, batch.fragments, batch.fragment_counts)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Collect predictions and labels for metrics calculation
            all_train_preds.extend(output.detach().cpu().numpy())
            all_train_labels.extend(batch.y.cpu().numpy())

        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        all_val_preds = []
        all_val_labels = []
        val_total_loss = 0

        with torch.no_grad():
            #for batch in test_loader:
            #for batch in batched_test_data:
            for batch in test_loader:
                batch = batch_path_data(batch)
                batch = batch.to(args.device)
                #print("\n===Test Batches: ")
                #print_batch_contents(batch)

                # Forward pass
                #output = model(batch.x, batch.edge_index, batch.edge_attr, batch.candidate_paths, batch.fragments, batch.fragment_counts, batch.batch)
                output = model(batch.x, batch.edge_index, batch.edge_attr, batch.candidate_paths, batch.splice_sources, batch.splice_targets, batch.abundance, batch.fragments, batch.fragment_counts)
                val_loss = criterion(output, batch.y)
                val_total_loss += val_loss.item()
                #print("val_total_loss: ", val_total_loss)

                # Collect predictions and labels for metrics calculation
                all_val_preds.extend(output.cpu().numpy())
                all_val_labels.extend(batch.y.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_total_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Step the scheduler with the validation loss
        #scheduler.step(val_loss)
        scheduler.step()

        if (epoch + 1) % args.plot_interval == 0 or epoch == args.num_epochs-1:
            plot_all_losses(train_losses, val_losses, start_epoch+1, args.runID)

        # Save the best model based on the lowest validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0

            val_auroc = roc_auc_score(all_val_labels, all_val_preds)
            best_auroc = val_auroc

            val_precision = precision_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), zero_division=0)
            val_recall = recall_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), zero_division=0)

            val_precision_curve, val_recall_curve, _ = precision_recall_curve(all_val_labels, all_val_preds)
            val_pr_auc = auc(val_recall_curve, val_precision_curve)

            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, "
                  f"Val Recall: {val_recall:.4f}, Val AUROC: {val_auroc:.4f}, Val PR AUC: {val_pr_auc:.4f}")

            #print(train_losses)
            #print(val_losses)
            model_save_path = f"{args.checkpoint_dir}/checkpoints-{args.runID}_ValLoss={val_loss:.4f}_epoch={epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(), #'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'loss_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
            }, model_save_path)

            print(f"New best model saved to {model_save_path} with Val Loss={best_loss:.4f}, Val AUROC={best_auroc:.4f}")
            
            # Store predictions and labels of the best model
            best_train_preds = all_train_preds
            best_train_labels = all_train_labels
            best_val_preds = all_val_preds
            best_val_labels = all_val_labels
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            plot_all_losses(train_losses, val_losses, start_epoch+1, args.runID)
            break

    return best_auroc, best_train_preds, best_train_labels, best_val_preds, best_val_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Path Scoring Model Configuration")
    
    # Model parameters
    parser.add_argument('--num_node_features', type=int, default=4, help='Number of node features')
    parser.add_argument('--num_edge_features', type=int, default=2, help='Number of edge features')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_gat_layers', type=int, default=1, help='Number of GAT layers')
    parser.add_argument('--hidden_dim_LSTM', type=int, default=64, help='Hidden dimension size of LSTM')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load a previous checkpoint')
    parser.add_argument('--load_pretrain', action='store_true', help='Whether to load a pretrained checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--plot_interval', type=int, default=50, help='Interval for plotting losses')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default="polyester_sim", help='Dataset name')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--sample_prefix', type=str, default='polyester_test1', help='Polyester test identifier')
    parser.add_argument('--samples', nargs='*', default=None, help='List of samples to load')
    parser.add_argument('--train_samples', nargs='*', default=None, help='List of training samples to load')
    parser.add_argument('--test_samples', nargs='*', default=None, help='List of testing samples to load')
    parser.add_argument('--train_prefix', type=str, default=None, help='Train prefix(.train)')
    parser.add_argument('--test_prefix', type=str, default=None, help='Test prefix')
    
    # Run parameters
    parser.add_argument('--runID', type=str, default='run-identifier', help='Run identifier')
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set default values for input_dir, checkpoint_dir, samples, train_prefix, and test_prefix if not provided
    if args.input_dir is None:
        args.input_dir = f"/storage/home/qzs23/work/PathEm/data/aletsch-8/{args.dataset}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"/storage/work/qzs23/PathEm/checkpoints/{args.sample_prefix}"
    #if args.samples is None:
        #args.samples = [f'{args.sample_prefix}_ensembl_1', f'{args.sample_prefix}_ensembl_2', f'{args.sample_prefix}_ensembl_3']
    if args.train_samples is None:
        args.train_samples = args.samples
    if args.test_samples is None:
        args.test_samples = args.samples
    if args.sample_prefix is not None:
        args.runID = f"{args.sample_prefix}-{args.runID}"
    if args.train_prefix is None:
        args.train_prefix = f"{args.dataset}.train"
    if args.test_prefix is None:
        args.test_prefix = f"{args.dataset}.test"
    

    return args

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# Usage in your main script
if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== {current_date}")
    print(f"Device: {args.device}, data loading")
    
    # Your data loading and processing code here
    train_node_df, train_edge_df, train_path_df, train_phasing_df = load_data(args.input_dir, args.train_prefix, args.train_samples)
    test_node_df, test_edge_df, test_path_df, test_phasing_df = load_data(args.input_dir, args.test_prefix, args.test_samples)
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== {current_date}, data processing")

    # Process graphs
    train_data_list = []
    for graph_id in train_node_df['graph_id'].unique():
        data = process_graph_to_data(train_node_df, train_edge_df, train_path_df, train_phasing_df, graph_id)
        if data is not None:
            train_data_list.append(data)
    print("Total #train graphs: ", len(train_data_list))
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== {current_date}, finish preprocessing training data")


    test_data_list = []
    for graph_id in test_node_df['graph_id'].unique():
        data = process_graph_to_data(test_node_df, test_edge_df, test_path_df, test_phasing_df, graph_id)
        if data is not None:
            test_data_list.append(data)
    
    print("Total #test graphs: ", len(test_data_list))
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== {current_date}, finish preprocessing testing data")
    
    # Create data loaders
    train_loader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate benchmark
    print("Training:")
    precision_train_bench, recall_train_bench, fpr_train_bench, tpr_train_bench, roc_auc_train_bench, pr_auc_train_bench = evaluate_benchmark(train_data_list)
    print("Testing:")
    precision_test_bench, recall_test_bench, fpr_test_bench, tpr_test_bench, roc_auc_test_bench, pr_auc_test_bench = evaluate_benchmark(test_data_list)
    
    print(f"Training Benchmark: ROC AUC = {roc_auc_train_bench:.4f}, PR AUC = {pr_auc_train_bench:.4f}")
    print(f"Testing Benchmark: ROC AUC = {roc_auc_test_bench:.4f}, PR AUC = {pr_auc_test_bench:.4f}")
    
    # Create and train model
    model = GPUEfficientPathScoringModel(args.num_node_features, args.num_edge_features, args.hidden_dim, args.num_heads, args.hidden_dim_LSTM, args.num_gat_layers)
    best_auroc, best_train_preds, best_train_labels, best_val_preds, best_val_labels = train_and_evaluate(
        model, train_loader, test_loader, args
    )
    
    # Plot results
    plot_auc_benchmark(
        best_train_labels, best_train_preds,
        best_val_labels, best_val_preds,
        (precision_train_bench, recall_train_bench, fpr_train_bench, tpr_train_bench, roc_auc_train_bench, pr_auc_train_bench),
        (precision_test_bench, recall_test_bench, fpr_test_bench, tpr_test_bench, roc_auc_test_bench, pr_auc_test_bench),
        args.runID
    )
    
    print(f"Best Test AUROC: {best_auroc:.4f}")
