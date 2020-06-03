import sys
import warnings
from random import sample, random, randint
warnings.filterwarnings("ignore")
from tensorflow.python.keras.models import load_model
from collections import deque
import gym
import numpy as np
from keras import Sequential
from keras.activations import relu, linear
from keras.layers import Dense
from keras.optimizers import adam


class DQNAgent:
    def __init__(
            self,
            state_space_dim,
            action_space_dim,
            learning_rate=10e-4,
            discount_factor=0.99,
            epsilon=1.0,
            memory_size=1000000,
            batch_size=64,
            hidden_layer_sizes=(80, 64)
    ):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.buffer = deque(maxlen=memory_size)
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.learning_rate = learning_rate
        self.steps_trained = 0
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model = self.build_nn()

    def build_nn(self):
        nn = Sequential()
        nn.add(Dense(self.hidden_layer_sizes[0], input_dim=self.state_space_dim, activation=relu))
        nn.add(Dense(self.hidden_layer_sizes[1], activation=relu))
        nn.add(Dense(self.action_space_dim, activation=linear))
        nn.compile(loss='mse', optimizer=adam(learning_rate=self.learning_rate))
        return nn

    def get_best_action(self, state):
        return np.argmax(self.model.predict(np.reshape(state, (1, 8)))[0])

    def get_action(self, state):
        if self.epsilon > 0.1:
            self.epsilon *= 0.99

        if random() < self.epsilon:
            return randint(0, self.action_space_dim - 1)
        return self.get_best_action(state)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = sample(self.buffer, self.batch_size)
        states = np.array([info[0] for info in batch])
        actions = np.array([info[1] for info in batch])
        next_states = np.array([info[2] for info in batch])
        rewards = np.array([info[3] for info in batch])
        dones = np.array([info[4] for info in batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        updated_targets = rewards + self.discount_factor * \
            (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)

        targets = self.model.predict_on_batch(states)

        targets[[np.array(list(range(self.batch_size)))], [actions]] = updated_targets

        self.model.fit(states, targets, epochs=1, verbose=0)
        self.steps_trained += 1

    def remember_step(self, info):
        self.buffer.append(info)

    def model_str(self):
        return f"model_{self.hidden_layer_sizes[0]}_{self.hidden_layer_sizes[1]}"

    def save(self):
        self.model.save(self.model_str())

    def load(self, model_path):
        self.model = load_model(model_path)


def run_dqn_agent(dqn_agent, env, test=False):
    try:
        max_steps_count = 3000
        episodes_count = 10000
        rewards_window = deque(maxlen=100)

        for episode_index in range(episodes_count):
            state = env.reset()
            episode_reward = 0
            for i in range(max_steps_count):

                action = dqn_agent.get_best_action(state) if test else dqn_agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                episode_reward += reward

                if not test:
                    dqn_agent.remember_step((state, action, next_state, reward, done))
                    dqn_agent.train_step()
                else:
                    env.render()
                state = next_state

                if done:
                    rewards_window.append(episode_reward)
                    print("episode: {}, reward: {}, avg: {}".format(episode_index, episode_reward, sum(rewards_window) / len(rewards_window)))
                    if not test and episode_index % 100 == 0:
                        dqn_agent.save()
                        print("Model has been saved")
                    break
    except KeyboardInterrupt:
        dqn_agent.save()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "test"}:
        print("pass 'train' or 'test' as agument")
        exit(1)
    cmd = sys.argv[1]
    env = gym.make("LunarLander-v2")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn_agent = DQNAgent(input_dim, output_dim)
    if cmd == "train":
        run_dqn_agent(dqn_agent, env)
    else:
        run_dqn_agent(dqn_agent, env, True)


if __name__ == '__main__':
    main()
