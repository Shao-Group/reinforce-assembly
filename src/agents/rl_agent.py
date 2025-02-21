#TODO

class RLAgent:
    def __init__(
        self,
        gnn_model: nn.Module,
        optimizer: optim.Optimizer,
        gamma: float = 0.99
    ):
        self.model = gnn_model
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state, valid_actions):
        data = state["data"]
        edge_logits = self.model(data)  

        # Create a mask for valid actions
        # valid_mask is 1 for valid edges, 0 for invalid
        num_edges = edge_logits.shape[0]
        mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_logits.device)
        mask[valid_actions] = True

        # Large negative for invalid edges to effectively remove them from the softmax
        masked_logits = torch.where(mask, edge_logits, torch.tensor(float('-inf'), device=edge_logits.device))

        # Convert logits to probabilities
        probs = F.softmax(masked_logits, dim=-1)
        
        # Sample from the distribution
        dist = torch.distributions.Categorical(probs)
        action_edge_idx = dist.sample()
        log_prob = dist.log_prob(action_edge_idx)

        return action_edge_idx.item(), log_prob

    def update_policy(self, trajectory):
        """
        Implements REINFORCE update.
        trajectory is a list of tuples:
            [(state, action, reward, log_prob), ..., (terminal_state, None, ...)]
        We first compute discounted returns, then compute the policy gradient loss.
        """
        rewards = [tr[2] for tr in trajectory]
        log_probs = [tr[3] for tr in trajectory if tr[3] is not None]

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        # Sum of (-log_prob * return)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()