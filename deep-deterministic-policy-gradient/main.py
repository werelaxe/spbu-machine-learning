import gym
import tensorflow as tf
import numpy as np

from ddpg_agent import DDPGAgent
from env_wrappers import get_pendulum_env


def main():
    env = get_pendulum_env()

    max_action_val = env.action_space.high[0]
    min_action_val = env.action_space.low[0]

    agent = DDPGAgent(env.state_space_dim, env.action_space_dim, min_action_val, max_action_val, 512)

    episode_rewards = []
    episodes_count = 5000

    for episode_index in range(episodes_count):
        episode_reward = 0
        state = env.reset()
        while True:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = agent.get_action(tf_state)

            next_state, reward, done, _ = env.step(action)
            agent.remember_step((state, action, next_state, reward))
            agent.learn()
            agent.update_targets()

            episode_reward += reward

            if done:
                break
            state = next_state
        episode_rewards.append(episode_reward)
        print(f"Episode #{episode_index}, reward: {episode_reward}, avg: {np.mean(episode_rewards[-100:])}")


if __name__ == '__main__':
    main()
