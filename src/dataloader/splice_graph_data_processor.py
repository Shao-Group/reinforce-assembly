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
        
        # Feature columns for nodes and edges
        # self.node_feature_cols = [
        #     'weight', 'length', 'maxcov', 'stddev', 'indel_sum_cov', 
        #     'indel_ratio', 'left_indel', 'right_indel'
        # ]

        self.node_feature_cols = [
            'weight', 'length', 'maxcov', 'stddev'
        ]

        self.edge_feature_cols = ['weight', 'length']
        
        # Load data frames
        self.node_df, self.edge_df, self.path_df, self.phasing_df = self._load_data()
        
        # Initialize and fit scalers if needed
        if normalize_features:
            self.node_scaler = StandardScaler()
            self.edge_scaler = StandardScaler()
            self._fit_scalers()
        
        # Filter valid graph IDs
        self.valid_graph_ids = self._filter_valid_graphs()
    
    def _load_data(self):
        """Load data files and apply sample filtering if needed."""
        # File paths
        node_file = os.path.join(self.input_dir, f'{self.prefix}.node.csv')
        edge_file = os.path.join(self.input_dir, f'{self.prefix}.edge.csv')
        path_file = os.path.join(self.input_dir, f'{self.prefix}.path.label.csv')
        phasing_file = os.path.join(self.input_dir, f'{self.prefix}.phasing.csv')
        
        # Define dtypes for efficiency
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
    
    def _fit_scalers(self):
        """Fit scalers to the node and edge features."""
        self.node_scaler.fit(self.node_df[self.node_feature_cols].values)
        self.edge_scaler.fit(self.edge_df[self.edge_feature_cols].values)
    
    def _filter_valid_graphs(self):
        """Filter graphs that have sufficient nodes, edges, and ground truth paths."""
        valid_ids = []
        all_graph_ids = self.node_df['graph_id'].unique()
        
        for graph_id in all_graph_ids:
            nodes = self.node_df[self.node_df['graph_id'] == graph_id]
            edges = self.edge_df[self.edge_df['graph_id'] == graph_id]
            paths = self.path_df[(self.path_df['graph_id'] == graph_id) & 
                                (self.path_df['label'] == 1)]
            
            # Valid if has sufficient nodes, edges, and at least one ground truth
            if len(nodes) >= 5 and len(edges) > 4 and len(paths) > 0:
                valid_ids.append(graph_id)
        
        return valid_ids
    
    def process_graph(self, graph_id):
        """
        Process a single graph into format required for reinforcement learning.
        
        Args:
            graph_id (str): ID of the graph to process
            
        Returns:
            data (Data): PyTorch Geometric Data object
            ground_truths (list): List of ground truth transcript paths
            sources (list): List of source node IDs
            sinks (list): List of sink node IDs
            node_id_map (dict): Mapping from original to contiguous node IDs
            reverse_node_id_map (dict): Mapping from contiguous to original node IDs
        """
        # Filter data for this graph
        nodes = self.node_df[self.node_df['graph_id'] == graph_id]
        edges = self.edge_df[self.edge_df['graph_id'] == graph_id]
        phasing = self.phasing_df[self.phasing_df['graph_id'] == graph_id]
        
        # Create node ID mapping (original ID → contiguous index)
        original_node_ids = nodes['node_id'].unique()
        node_id_map = {nid: i for i, nid in enumerate(original_node_ids)}
        reverse_node_id_map = {i: nid for nid, i in node_id_map.items()}
        
        # Extract and process node features
        node_features = []
        for nid in original_node_ids:
            node_row = nodes[nodes['node_id'] == nid].iloc[0]
            node_features.append([node_row[col] for col in self.node_feature_cols])
        
        # Normalize node features if requested
        node_features = np.array(node_features)
        if self.normalize_features:
            node_features = self.node_scaler.transform(node_features)
        
        # Convert edge source/target to use contiguous IDs
        edge_source = [node_id_map[s] for s in edges['source']]
        edge_target = [node_id_map[t] for t in edges['target']]
        
        # Create edge_index tensor [2, num_edges]
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        
        # Extract and normalize edge features
        edge_features = edges[self.edge_feature_cols].values
        if self.normalize_features:
            edge_features = self.edge_scaler.transform(edge_features)
        
        # Calculate phasing path coverage
        num_nodes = len(node_id_map)
        node_phasing_coverage = torch.zeros(num_nodes, 1, dtype=torch.float)
        edge_phasing_coverage = torch.zeros(len(edge_source), 1, dtype=torch.float)
        
        # Create edge lookup for efficient updates
        edge_lookup = {}
        for i, (s, t) in enumerate(zip(edge_source, edge_target)):
            edge_lookup[(s, t)] = i
        
        # Process phasing paths
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
        
        # Process ground truth paths
        ground_truths, sources, sinks = self._process_ground_truths(graph_id, node_id_map)
        
        return data, ground_truths, sources, sinks, node_id_map, reverse_node_id_map
    
    def _process_ground_truths(self, graph_id, node_id_map):
        """
        Extract ground truth transcript paths for the given graph.
        
        Args:
            graph_id (str): Graph ID
            node_id_map (dict): Mapping from original node IDs to contiguous IDs
            
        Returns:
            ground_truth_paths (list): List of ground truth paths as node sequences
            sources (list): Source node IDs (start of transcripts)
            sinks (list): Sink node IDs (end of transcripts) 
        """
        # Filter to labeled paths (ground truths)
        filtered_df = self.path_df[(self.path_df['graph_id'] == graph_id) & 
                                   (self.path_df['label'] == 1)]
        
        # Sort by abundance and keep top paths
        top_paths = filtered_df.sort_values(by='abundance', ascending=False
                                           ).head(self.num_ground_truths)
        
        ground_truth_paths = []
        sources = set()
        sinks = set()
        
        for _, row in top_paths.iterrows():
            try:
                # Parse node sequence and map to new IDs
                seq_str = row['node_sequence']
                original_nodes = [int(node) for node in seq_str.split(',')]
                node_list = [node_id_map.get(node, -1) for node in original_nodes]
                
                # Skip invalid paths
                if -1 in node_list:
                    continue
                
                ground_truth_paths.append(node_list)
                sources.add(node_list[0])     # First node is a source
                sinks.add(node_list[-1])      # Last node is a sink
            except:
                continue
        
        return ground_truth_paths, list(sources), list(sinks)
    
    def get_all_graph_ids(self):
        """Return list of all valid graph IDs."""
        return self.valid_graph_ids


class SpliceGraphDataset(Dataset):
    def __init__(self, data_processor):
        """
        PyTorch Geometric Dataset for splice graphs used in RL.
        
        Args:
            data_processor (SpliceGraphDataProcessor): Processor for splice graph data
        """
        super().__init__()
        self.data_processor = data_processor
        self.graph_ids = data_processor.get_all_graph_ids()
    
    def len(self):
        """Return number of graphs in the dataset."""
        return len(self.graph_ids)
    
    def get(self, idx):
        """
        Get a graph by index, with all information needed for RL environment.
        
        Args:
            idx (int): Index in the dataset
            
        Returns:
            data_dict (dict): Dictionary containing:
                - data: PyG Data object with node/edge features
                - ground_truths: List of ground truth paths
                - sources: List of source node IDs
                - sinks: List of sink node IDs
                - node_id_map: Original to contiguous ID mapping
                - reverse_node_id_map: Contiguous to original ID mapping
        """
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