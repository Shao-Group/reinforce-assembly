# TODO: start_node =  
# TODO: end_node = 
# TODO: Provide a ground_truth_path (list of edges or nodes) used for reward shaping:

# TODO: env = SpliceGraphEnv(
#     data=data_obj,
#     start_node=start_node,
#     end_node=end_node,
#     ground_truth_path=ground_truth_path,
#     max_steps=100  # or any suitable limit
# )

# num_node_features = data_obj.x.size(-1)
# num_edge_features = data_obj.edge_attr.size(-1)

# gnn_model = GNNForRLAgent(
#     num_node_features=num_node_features,
#     num_edge_features=num_edge_features,
#     hidden_dim=64,  
#     num_heads=4,
#     num_gat_layers=3
# )

# optimizer = optim.Adam(gnn_model.parameters(), lr=1e-3)

# agent = RLAgent(
#     gnn_model=gnn_model,
#     optimizer=optimizer,
#     gamma=0.99
# )

# train_agent(env, agent, num_episodes=50)

if __name__ == '__main__':
    pass
