import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


class SpliceGraphDataProcessor:
    def __init__(self, node_file_path, edge_file_path, ground_truth_file_path, phasing_file_path, samples=None, simplify=True):
        self.node_file_path = node_file_path
        self.edge_file_path = edge_file_path
        self.ground_truth_file_path = ground_truth_file_path
        self.phasing_file_path = phasing_file_path
        self.samples = samples
        self.simplify = simplify
        self.graph_ids = None

        self.node_df, self.edge_df, self.ground_truth_df, self.phasing_df = self._load_data()
    
    def _load_data(self):
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
        
        # Load CSVs
        node_df = pd.read_csv(self.node_file_path, dtype=node_dtypes)
        edge_df = pd.read_csv(self.edge_file_path, dtype=edge_dtypes)
        ground_truth_df = pd.read_csv(self.ground_truth_file_path, dtype=path_dtypes)
        phasing_df = pd.read_csv(self.phasing_file_path, dtype=phasing_dtypes)
        
        # Apply sample filtering if specified
        if self.samples:
            node_df = node_df[node_df['sample'].isin(self.samples)]
            edge_df = edge_df[edge_df['sample'].isin(self.samples)]
            ground_truth_df = ground_truth_df[ground_truth_df['sample'].isin(self.samples)]
            phasing_df = phasing_df[phasing_df['sample'].isin(self.samples)]
        
        if self.simplify:
            node_df = node_df.drop(['chr', 'start_pos', 'end_pos', 'extPathSupport', 'maxcov', 
                'stddev', 'indel_sum_cov', 'indel_ratio', 'left_indel', 'right_indel'], axis=1)
            edge_df = edge_df.drop(['chr', 'start_pos', 'end_pos',
                'extPathSupport'], axis=1)
            ground_truth_df = ground_truth_df.drop(['chr', 'path_id', 'splice_source',
                'splice_target', 'abundance', 'GroundTruthID'], axis=1)
            phasing_df = phasing_df.drop(['chr', 'path_id'], axis=1)

            counts = ground_truth_df['graph_id'].value_counts()
            graph_ids_numgt_ge_4 = set(counts[counts >= 4].index.tolist())

            phasing_columns_set = set(phasing_df['graph_id'])
            node_df = node_df[node_df['graph_id'].isin(graph_ids_numgt_ge_4 & phasing_columns_set)]
            edge_df = edge_df[edge_df['graph_id'].isin(graph_ids_numgt_ge_4 & phasing_columns_set)]
            ground_truth_df = ground_truth_df[ground_truth_df['graph_id'].isin(graph_ids_numgt_ge_4 & phasing_columns_set)]
            phasing_df = phasing_df[phasing_df['graph_id'].isin(graph_ids_numgt_ge_4)]

            self.graph_ids = node_df['graph_id'].unique()
        
        return node_df, edge_df, ground_truth_df, phasing_df
    
    def _get_dfs(self):
        return self.node_df, self.edge_df, self.ground_truth_df, self.phasing_df
    
    def process_graph(self, graph_id):
        nodes = self.node_df[self.node_df['graph_id'] == graph_id]
        edges = self.edge_df[self.edge_df['graph_id'] == graph_id]
        ground_truths = self.ground_truth_df[self.ground_truth_df['graph_id'] == graph_id]
        phasing = self.phasing_df[self.phasing_df['graph_id'] == graph_id]
        
        # Create node ID mapping (original ID → contiguous index)
        original_node_ids = nodes['node_id'].unique()
        node_id_map = {nid: i for i, nid in enumerate(original_node_ids)}
        reverse_node_id_map = {i: nid for nid, i in node_id_map.items()}
        
        # Extract and process node features
        node_features = []
        for nid in original_node_ids:
            node_row = nodes[nodes['node_id'] == nid].iloc[0]
            node_features.append([node_row[col] for col in ['weight', 'length']])
    
        # Convert edge source/target to use contiguous IDs
        edge_source = [node_id_map[s] for s in edges['source']]
        edge_target = [node_id_map[t] for t in edges['target']]
        
        # Create edge_index tensor [2, num_edges]
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        
        edge_features = edges[['weight', 'length']].values
        
        # Calculate phasing path coverage
        num_nodes = len(node_id_map)
        node_phasing_coverage = torch.zeros(num_nodes, 1, dtype=torch.float)
        edge_phasing_coverage = torch.zeros(len(edge_source), 1, dtype=torch.float)
        
        edge_lookup = {}
        for i, (s, t) in enumerate(zip(edge_source, edge_target)):
            edge_lookup[(s, t)] = i
        
        for _, row in phasing.iterrows():
            try:
                node_seq_str = row['node_sequence']
                count_value = row['count']
                
                # Map original node IDs to contiguous indices
                original_nodes = [int(n) for n in node_seq_str.split(',')]
                node_list = [node_id_map.get(n, -1) for n in original_nodes]
                
                # Skip invalid paths
                if -1 in node_list:
                    continue
                
                # Update node phasing coverage
                for nid in node_list:
                    node_phasing_coverage[nid, 0] += count_value
                
                # Update edge phasing coverage
                for i in range(len(node_list) - 1):
                    s, t = node_list[i], node_list[i + 1]
                    if (s, t) in edge_lookup:
                        edge_idx = edge_lookup[(s, t)]
                        edge_phasing_coverage[edge_idx, 0] += count_value
            except:
                continue
        
        # Initialize is_in_partial_path features (will be modified during RL)
        node_in_path = torch.zeros(num_nodes, 1, dtype=torch.float)
        edge_in_path = torch.zeros(len(edge_source), 1, dtype=torch.float)
        
        # Concatenate features
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        
        node_features = torch.cat([node_features, node_phasing_coverage, node_in_path], dim=1)
        edge_features = torch.cat([edge_features, edge_phasing_coverage, edge_in_path], dim=1)
        
        # Create PyG Data object for the GNN
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        ground_truth_paths = []
        sources = set()
        sinks = set()

        for _, row in ground_truths.iterrows():
            try:
                node_seq_str = row['node_sequence']
                
                # Map original node IDs to contiguous indices
                original_nodes = [int(n) for n in node_seq_str.split(',')]
                node_list = [node_id_map.get(n, -1) for n in original_nodes]
                
                # Skip invalid paths
                if -1 in node_list:
                    continue

                ground_truth_paths.append(node_list)
                sources.add(node_list[0])
                sinks.add(node_list[-1])
        
            except:
                continue
        
        return data, ground_truth_paths, sources, sinks, node_id_map, reverse_node_id_map


class SpliceGraphDataset(Dataset):
    def __init__(self, data_processor):
        """
        PyTorch Geometric Dataset for splice graphs used in RL.
        
        Args:
            data_processor (SpliceGraphDataProcessor): Processor for splice graph data
        """
        super().__init__()
        self.data_processor = data_processor
        self.graph_ids = data_processor.graph_ids
    
    def len(self):
        """Return number of graphs in the dataset."""
        return len(self.graph_ids)
    
    def get(self, idx):
        if idx >= len(self.graph_ids):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.graph_ids)} graphs")
        
        graph_id = self.graph_ids[idx]
        data, ground_truths, sources, sinks, node_id_map, reverse_node_id_map = (
            self.data_processor.process_graph(graph_id)
        )
        
        # Return as a dictionary with all components needed for RL
        return {
            'data': data,
            'ground_truths': ground_truths,
            'sources': sources,
            'sinks': sinks,
            'node_id_map': node_id_map,
            'reverse_node_id_map': reverse_node_id_map,
            'graph_id': graph_id
        }