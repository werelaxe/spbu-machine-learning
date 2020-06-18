import os

import tensorflow as tf
import numpy as np

from ddpg_agent import DDPGAgent
from env_wrappers import get_pendulum_env


MODEL_PATH = "model"


def main():
    env = get_pendulum_env()

    max_action_val = env.action_space.high[0]
    min_action_val = env.action_space.low[0]

    agent = DDPGAgent(
        env.state_space_dim,
        env.action_space_dim,
        min_action_val,
        max_action_val,
        hidden_layer_size=512,
        path_to_load=MODEL_PATH
    )

    episode_rewards = []
    episodes_count = 5000

    render = False
    for episode_index in range(episodes_count):
        try:
            episode_reward = 0
            state = env.reset()
            while True:
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action = agent.get_action(tf_state)

                next_state, reward, done, _ = env.step(action)
                agent.remember_step((state, action, next_state, reward))
                if render:
                    env.render()
                agent.learn()
                agent.update_targets()

                episode_reward += reward

                if done:
                    break
                state = next_state
            episode_rewards.append(episode_reward)
            print(f"Episode #{episode_index}, reward: {episode_reward}, avg: {np.mean(episode_rewards[-100:])}")
            if episode_index % 10 == 0:
                agent.save(MODEL_PATH)
        except KeyboardInterrupt:
            agent.save(MODEL_PATH)
            cmd = input("Input: ")
            if cmd == "1":
                render = True
            elif cmd == "0":
                render = False


if __name__ == '__main__':
    main()
