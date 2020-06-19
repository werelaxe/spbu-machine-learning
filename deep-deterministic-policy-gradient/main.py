import os

import tensorflow as tf
import numpy as np
from dm_control import viewer

from ddpg_agent import DDPGAgent
from env_wrappers import get_swimmer6_env, get_state

MODEL_PATH = "model"


def view_render(env, agent):
    def random_policy(time_step):
        return agent.get_best_action(get_state(time_step.observation))
    viewer.launch(env.env, policy=random_policy)


def main():
    env = get_swimmer6_env()

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
    episodes_count = 50000

    for episode_index in range(episodes_count):
        try:
            episode_reward = 0
            state = env.reset()
            current_reward = None
            while True:
                action = agent.get_action(state)

                next_state, reward, done, _ = env.step(action)
                if current_reward is None:
                    current_reward = reward
                    continue

                reward_diff = reward - current_reward
                current_reward = reward

                agent.remember_step((state, action, next_state, reward_diff))
                agent.learn()
                agent.update_targets()

                episode_reward += reward_diff

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
                view_render(env, agent)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
