import os

import tensorflow as tf
import numpy as np

from actor import Actor
from buffer import Buffer
from critic import Critic
from noise import GaussianNoise


class DDPGAgent:
    def __init__(
            self,
            state_space_dim,
            action_space_dim,
            min_action_val,
            max_action_val,
            hidden_layer_size=512,
            gamma=0.99,
            tau=0.005,
            path_to_load=None
    ):
        self.gamma = gamma
        self.tau = tau
        self.min_action_val = min_action_val
        self.max_action_val = max_action_val
        self.buffer = Buffer(state_space_dim, action_space_dim)
        self.noise_generator = GaussianNoise(0., 0.2, action_space_dim)

        self.actor = Actor(state_space_dim, action_space_dim, max_action_val, hidden_layer_size)
        self.critic = Critic(state_space_dim, action_space_dim, hidden_layer_size)

        if path_to_load is not None:
            if os.path.exists(path_to_load + "_actor.h5") and \
                    os.path.exists(path_to_load + "_critic.h5"):
                self.load(path_to_load)

        self.target_actor = Actor(state_space_dim, action_space_dim, max_action_val, hidden_layer_size)
        self.target_critic = Critic(state_space_dim, action_space_dim, hidden_layer_size)

        self.target_actor.model.set_weights(self.actor.model.get_weights())
        self.target_critic.model.set_weights(self.critic.model.get_weights())

        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    @tf.function
    def _apply_gradients(self, states, actions, next_states, rewards):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.forward(next_states)
            y = tf.cast(rewards, tf.float32) + self.gamma * self.target_critic.forward(
                [next_states, target_actions])
            critic_value = self.critic.forward([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor.forward(states)
            critic_value = self.critic.forward([states, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables)
        )

    def learn(self):
        states, actions, next_states, rewards = self.buffer.sample()
        self._apply_gradients(states, actions, next_states, rewards)

    def remember_step(self, info):
        self.buffer.remember(info)

    def update_targets(self):
        new_weights = []
        target_variables = self.target_critic.model.weights
        for i, variable in enumerate(self.critic.model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic.model.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.model.weights
        for i, variable in enumerate(self.actor.model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor.model.set_weights(new_weights)

    def get_action(self, state):
        actions = tf.squeeze(self.actor.forward(state)).numpy() + self.noise_generator.get_noise()
        return np.clip(actions, self.min_action_val, self.max_action_val)

    def save(self, path):
        print(f"Model has been saved as '{path}'")
        self.actor.save(path)
        self.critic.save(path)

    def load(self, path):
        print(f"Model has been loaded from '{path}'")
        self.actor.load(path)
        self.critic.load(path)
