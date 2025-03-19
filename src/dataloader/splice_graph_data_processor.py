import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler


class SpliceGraphDataProcessor:
    def __init__(self, input_dir, prefix, samples=None, num_ground_truths=5, normalize_features=True):
        self.input_dir = input_dir
        self.prefix = prefix
        self.samples = samples
        self.num_ground_truths = num_ground_truths
        self.normalize_features = normalize_features

        self.node_feature_cols = [
            'weight', 'length', 'maxcov', 'stddev', 'indel_sum_cov', 
            'indel_ratio', 'left_indel', 'right_indel'
        ]
        self.edge_feature_cols = ['weight', 'length']

        self.node_df, self.edge_df, self.path_df, self.phasing_df = self._load_data()

        self.valid_graph_ids = self._filter_valid_graphs()

    def _load_data(self):
        node_file = os.path.join(self.input_dir, f'{self.prefix}.node.csv')
        edge_file = os.path.join(self.input_dir, f'{self.prefix}.edge.csv')
        path_file = os.path.join(self.input_dir, f'{self.prefix}.path.label.csv')
        phasing_file = os.path.join(self.input_dir, f'{self.prefix}.phasing.csv')

        node_dtypes = {
            "chr": str, "graph_id": str, "node_id": int, 
            "start_pos": np.uint64, "end_pos": np.uint64, 
            "weight": float, "length": int, "maxcov": float, 
            "stddev": float, "indel_sum_cov": float, 
            "indel_ratio": float, "left_indel": int, 
            "right_indel": int, "sample": str
        }
        edge_dtypes = {
            "chr": str, "graph_id": str, "source": int, "target": int, 
            "start_pos": np.uint64, "end_pos": np.uint64, 
            "weight": float, "length": int, "sample": str
        }
        path_dtypes = {
            "chr": str, "graph_id": str, "path_id": str, 
            "node_sequence": str, "splice_source": str, 
            "splice_target": str, "abundance": float, 
            "label": int, "sample": str
        }
        phasing_dtypes = {
            "chr": str, "graph_id": str, "path_id": str, 
            "node_sequence": str, "count": int, "sample": str
        }

        node_df = pd.read_csv(node_file, dtype=node_dtypes)
        edge_df = pd.read_csv(edge_file, dtype=edge_dtypes)
        path_df = pd.read_csv(path_file, dtype=path_dtypes)
        phasing_df = pd.read_csv(phasing_file, dtype=phasing_dtypes)
        
        # Apply sample filtering if specified
        if self.samples:
            node_df = node_df[node_df['sample'].isin(self.samples)]
            edge_df = edge_df[edge_df['sample'].isin(self.samples)]
            path_df = path_df[path_df['sample'].isin(self.samples)]
            phasing_df = phasing_df[phasing_df['sample'].isin(self.samples)]
        
        return node_df, edge_df, path_df, phasing_df
    
    def _filter_valid_graphs(self):
        valid_ids = []
        all_graph_ids = self.node_df['graph_id'].unique()
        
        for graph_id in all_graph_ids:
            nodes = self.node_df[self.node_df['graph_id'] == graph_id]
            edges = self.edge_df[self.edge_df['graph_id'] == graph_id]
            paths = self.path_df[(self.path_df['graph_id'] == graph_id) & 
                                (self.path_df['label'] == 1)]
            
            # Valid if has sufficient nodes, edges, and at least one ground truth
            if len(nodes) >= 5 and len(edges) > 4 and len(paths) > 5:
                valid_ids.append(graph_id)
        
        return valid_ids
    
    def process_graph(self, graph_id):
        nodes = self.node_df[self.node_df['graph_id'] == graph_id]
        edges = self.edge_df[self.edge_df['graph_id'] == graph_id]
        phasing = self.phasing_df[self.phasing_df['graph_id'] == graph_id]

        original_node_ids = nodes['node_id'].unique()
        
        node_features = []
        for nid in original_node_ids:
            node_row = nodes[nodes['node_id'] == nid].iloc[0]
            node_features.append([node_row[col] for col in self.node_feature_cols])