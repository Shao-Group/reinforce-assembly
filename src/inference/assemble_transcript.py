def assemble_transcript(env, agent):
    """
    Runs the environment from start to end using the trained agent,
    returns the list of edges visited and the corresponding node path.
    """
    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            env.step(-1)
            break

        action_idx, _ = agent.select_action(state, valid_actions)
        next_state, reward, done, _ = env.step(action_idx)
        state = next_state

    # The environment tracks visited edges in env.path_history.
    node_path = [env.start_node]
    for edge_idx in env.path_history:
        dst_node = env.edge_index[1][edge_idx].item()
        node_path.append(dst_node)

    return env.path_history, node_path

final_edge_path, final_node_path = assemble_transcript(env, agent)

print("Assembled transcript (edge indices):", final_edge_path)
print("Assembled transcript (node sequence):", final_node_path)
