#TODO

class SpliceGraphEnv:
    def __init__(
        self,
        data: Data,
        start_node: int,
        end_node: int,
        ground_truth_path: list,
        max_steps: int = 50,
        step_penalty: float = -0.01
    ):
       
        self.data = data
        self.start_node = start_node
        self.end_node = end_node
        self.ground_truth_path = ground_truth_path  
        self.max_steps = max_steps
        self.step_penalty = step_penalty

        self.edge_index = data.edge_index
        self.num_edges = self.edge_index.shape[1]

        self.current_node = None
        self.visited_edges = None
        self.steps_taken = 0
        self.done = False
        self.path_history = []  

        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.visited_edges = set()
        self.steps_taken = 0
        self.done = False
        self.path_history = []
        return self.get_state()

    def get_state(self):
        return {
            "current_node": self.current_node,
            "visited_edges": self.visited_edges,
            "steps_taken": self.steps_taken,
            "data": self.data
        }

    def get_valid_actions(self):
        valid_edges = []
        src_nodes = self.edge_index[0]  
        for e_idx in range(self.num_edges):
            if src_nodes[e_idx].item() == self.current_node and e_idx not in self.visited_edges:
                valid_edges.append(e_idx)
        return valid_edges

    def step(self, action_edge_idx):
        # If the edge is already visited or is invalid, we can penalize and end the episode
        valid_actions = self.get_valid_actions()
        if action_edge_idx not in valid_actions:
            # Invalid action chosen
            reward = -1.0
            self.done = True
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        # Mark edge as visited
        self.visited_edges.add(action_edge_idx)
        self.path_history.append(action_edge_idx)

        # Move to next node
        dst_node = self.edge_index[1][action_edge_idx].item()
        self.current_node = dst_node

        # Step penalty
        reward = self.step_penalty
        self.steps_taken += 1

        # Check if we reached the end node
        if self.current_node == self.end_node:
            # Compare assembled path to the ground truth
            final_transcript_reward = self._evaluate_transcript()
            reward += final_transcript_reward
            self.done = True

        # Check if maximum steps exceeded
        if self.steps_taken >= self.max_steps:
            # If we haven't reached end_node by now, we can finalize reward as well
            if not self.done:
                # Possibly some penalty for not finishing
                reward -= 0.5
                self.done = True

        next_state = self.get_state()
        return next_state, reward, self.done, {}

    def _evaluate_transcript(self):
        # TODO: If the nodes or edges match exactly, big reward.
        # Otherwise, partial match or some scoring. We assume ground_truth_path
        # is a list of edges for simplicity. If it's a list of nodes, adapt accordingly.
        if self.path_history == self.ground_truth_path:
            return 10.0  # perfect match reward
        else:
            return -0.2  # small negative if not perfect
