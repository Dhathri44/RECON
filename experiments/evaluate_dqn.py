
import numpy as np

def evaluate_dqn(agent, env, episodes=100):

    episode_rewards = []
    episode_psnr = []
    episode_ssim = []
    episode_bytes = []
    episode_delays = []

    action_counts = np.zeros(3, dtype=int)
    successful_episodes = 0

    for episode in range(episodes):

        state = env.reset()
        done = False
        total_reward = 0

        while not done:

            # Evaluation: no random exploration
            action = agent.select_action(
                state,
                evaluate=True
            )

            action_counts[action] += 1

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

        # These keys are examples.
        # Replace them with your actual environment's output.
        episode_psnr.append(info["psnr"])
        episode_ssim.append(info["ssim"])
        episode_bytes.append(info["bytes"])
        episode_delays.append(info["delay"])

        if info["recovery_success"]:
            successful_episodes += 1

    print("\n===== DQN EVALUATION =====")
    print("Average reward:", np.mean(episode_rewards))
    print("Average PSNR:", np.mean(episode_psnr))
    print("Average SSIM:", np.mean(episode_ssim))
    print("Average bytes:", np.mean(episode_bytes))
    print("Average delay:", np.mean(episode_delays))
    print(
        "Recovery success rate:",
        successful_episodes / episodes * 100,
        "%"
    )

    print("\nAction distribution:")
    for action in range(3):
        print(
            f"Action {action}:",
            action_counts[action],
            f"({action_counts[action] / action_counts.sum() * 100:.2f}%)"
        )
