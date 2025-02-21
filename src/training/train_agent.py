def train_agent(env: SpliceGraphEnv, agent: RLAgent, num_episodes: int = 500):
    for episode in range(num_episodes):
        state = env.reset()

        # Storage for trajectory
        # (state, action, reward, log_prob)
        trajectory = []

        while True:
            valid_actions = env.get_valid_actions()

            # Edge case: if no valid actions remain, break
            if len(valid_actions) == 0:
                _, reward, done, _ = env.step(-1)  # triggers invalid action
                # Add final step to trajectory
                trajectory.append((state, -1, reward, None))
                break

            # Agent selects an action
            action_edge_idx, log_prob = agent.select_action(state, valid_actions)
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(action_edge_idx)

            # Store in trajectory
            trajectory.append((state, action_edge_idx, reward, log_prob))

            state = next_state

            if done:
                break

        # Update policy at the end of the episode
        agent.update_policy(trajectory)

        # Logging / print out
        total_reward = sum([x[2] for x in trajectory])
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.3f}")

    print("Trainng done")