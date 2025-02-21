import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

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


def process_input_graph_to_input_data(node_df, edge_df, phasing_df, graph_id):

    nodes = node_df[node_df['graph_id'] == graph_id]
    edges = edge_df[edge_df['graph_id'] == graph_id]
    phasing = phasing_df[phasing_df['graph_id'] == graph_id]
    
    if len(nodes) < 5:
        return None
    if len(edges) == 0:
        return None

    node_feature_cols = ['extPathSupport', 'weight', 'length', 'maxcov', 'stddev', 'indel_sum_cov','indel_ratio','left_indel','right_indel']
    edge_feature_cols = ['extPathSupport', 'weight', 'length']

    # Define torch tensors with custom features for nodes and edges
    node_features = torch.tensor(nodes[node_feature_cols].values, dtype=torch.float)
    edge_features = torch.tensor(edges[edge_feature_cols].values, dtype=torch.float)

    # Tensor for the Data object to maintain links
    edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)

    num_nodes = len(nodes)
    num_edges = len(edges)

    # Initialize coverage with zeros
    node_phasing_coverage = torch.zeros(num_nodes, dtype=torch.float)
    edge_phasing_coverage = torch.zeros(num_edges, dtype=torch.float)

    # Build a lookup from node_id -> row index in 'nodes'
    node_ids = nodes['node_id'].values
    node_index_lookup = {nid: i for i, nid in enumerate(node_ids)}

    # Build a lookup for (source, target) -> row index in 'edges'
    edge_lookup = {}
    for i, e_row in edges.iterrows():
        s, t = e_row['source'], e_row['target']
        edge_lookup[(s, t)] = i

    # Aggregate phasing coverage
    for _, row in phasing.iterrows():
        node_seq_str = row.get('node_sequence', None)
        if not isinstance(node_seq_str, str):
            continue  # skip invalid or empty sequences
        
        count_value = row.get('count', 0)
        node_list = list(map(int, node_seq_str.split(',')))  # e.g. "0,1,2" -> [0,1,2]

        # Update node coverage
        for nid in node_list:
            if nid in node_index_lookup:
                node_phasing_coverage[node_index_lookup[nid]] += count_value

        # Update edge coverage (for consecutive pairs)
        for s, t in zip(node_list, node_list[1:]):
            if (s, t) in edge_lookup:
                edge_idx = edge_lookup[(s, t)]
                edge_phasing_coverage[edge_idx] += count_value

    node_phasing_coverage = node_phasing_coverage.view(-1, 1)  # shape [num_nodes, 1]
    edge_phasing_coverage = edge_phasing_coverage.view(-1, 1)  # shape [num_edges, 1]

    node_features = torch.cat([node_features, node_phasing_coverage], dim=1)
    edge_features = torch.cat([edge_features, edge_phasing_coverage], dim=1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features
    )

    return data
