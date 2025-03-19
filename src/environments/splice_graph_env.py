from typing import List, Dict, Tuple, Optional, Union, Any
import random
from torch_geometric.data import Data
import torch
import Levenshtein


class SpliceGraphEnv:
    def __init__(self, data_sample: Dict[str, Any]):
        self.graph_data = data_sample['data']
        self.ground_truths = data_sample['ground_truths']
        self.sources = data_sample['sources']
        self.sinks = data_sample['sinks']
        self.node_id_map = data_sample['node_id_map']
        self.reverse_node_id_map = data_sample['reverse_node_id_map']
        self.graph_id = data_sample['graph_id']

        # TODO: Reward part is to be modified later:
        # step penalty implied shorter transcripts are better which is not true
        # change the other rewards by researching more
        self.step_penalty = -1
        self.match_reward = 100
        self.sink_no_match_reward = 5
        self.invalid_termination_reward = -20

        # This can be changed.
        self.max_runs = 5

        # State Tracking variables
        self.current_node = None
        self.current_partial_path = None
        self.paths_collected = None

        self.run_number = None
        self.steps_taken_in_current_run = None

        self.done = None
        self.info = {}

        self.working_graph = None
        self.total_reward = 0
        
    def _copy_graph(self, graph: Data) -> Data:
        new_graph = Data()
        new_graph.x = graph.x.clone()
        new_graph.edge_index = graph.edge_index.clone()
        new_graph.edge_attr = graph.edge_attr.clone()
        
        return new_graph
    
    def _find_edge_idx(self, src_idx, dst_idx):
        for i in range(self.working_graph.edge_index.shape[1]):
            if (self.working_graph.edge_index[0, i] == src_idx) and (self.working_graph.edge_index[1, i] == dst_idx):
                return i
        return None
    
    def _update_node_in_path(self, node_idx):
        self.working_graph.x[node_idx, -1] += 1.0

    def _update_edge_in_path(self, src_idx, dst_idx):
        edge_idx = self._find_edge_idx(src_idx, dst_idx)
        if edge_idx is not None:
            self.working_graph.edge_attr[edge_idx, -1] += 1.0

    def _check_path_match(self, path):
        for gt in self.ground_truths:
            if len(path) == len(gt) and all(int(a) == int(b) for a, b in zip(path, gt)):
                return True
        return False
    
    def _start_new_run(self):
        self.current_partial_path = []
        self.current_node = random.choice(self.sources)
        self.current_partial_path.append(self.current_node)
        self.steps_taken_in_current_run = 0
        self._update_node_in_path(self.current_node)

    def reset(self):
        self.current_partial_path = []
        self.paths_collected = []

        self.run_number = 1
        self.steps_taken_in_current_run = 0

        self.done = False
        self.info = {}
        self.total_reward = 0

        self.working_graph = self._copy_graph(self.graph_data)

        self._start_new_run()

        return self.working_graph
    
    # TODO: create adjacency list for storing valid actions; this is currently very inefficient
    def get_valid_actions(self):
        valid_actions = []
        edge_index = self.working_graph.edge_index

        for i in range(edge_index.shape[1]):
            if edge_index[0, i] == self.current_node:
                next_node = edge_index[1, i].item()
                valid_actions.append(next_node)
        
        return valid_actions
    
    def _evaluate_path_reward(self, path):
        path_reward = 0
        
        # Check if path terminates at a sink
        if path[-1] in self.sinks:
            # Check if path matches ground truth
            if self._check_path_match(path):
                path_reward += self.match_reward
                self.info[f'run_{self.run_number}_match'] = True
            else:
                path_reward += self.sink_no_match_reward
                self.info[f'run_{self.run_number}_match'] = False
        else:
            # Path terminated without reaching sink
            path_reward += self.invalid_termination_reward
            self.info[f'run_{self.run_number}_invalid_termination'] = True
            
        return path_reward
    
    # Reward at the end (empty for now)
    def _finalize_episode(self):
        pass
    
    def step(self, action):
        prev_node = self.current_node
        self.current_node = action
        self.current_partial_path.append(self.current_node)
        self.steps_taken_in_current_run += 1

        self._update_node_in_path(action)
        self._update_edge_in_path(prev_node, action)

        step_reward = self.step_penalty
        self.total_reward += step_reward

        if action in self.sinks or not self.get_valid_actions():
            self.paths_collected.append(self.current_partial_path.copy())
            path_reward = self._evaluate_path_reward(self.current_partial_path)
            self.total_reward += path_reward

            if self.run_number == self.max_runs:
                self._finalize_episode()
                self.done = True
            else:
                self.run_number += 1
                self._start_new_run()
        
        return self.working_graph, self.total_reward, self.done, self.info

    def render(self):
        """Display current state information."""
        print(f"Run: {self.run_number}/{self.max_runs}")
        print(f"Current node: {self.current_node}")
        print(f"Current path: {self.current_partial_path}")
        print(f"Valid actions: {self.get_valid_actions()}")
        print(f"Paths collected: {len(self.paths_collected)}/{self.max_runs}")
        print(f"Total reward: {self.total_reward}")