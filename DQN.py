import random
import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam


class DQN:
    memory = list()

    def __init__(self, learning_rate, epsilon, epsilon_decay, gamma, tau, batch_size, buffer_size,
                 min_memory_size):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_memory_size = min_memory_size
        self.model = None
        self.target_model = None

    # create DQN model
    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(self.learning_rate))

        return model

    # storing agent experience replay memory
    def store_experience(self, experience: list):
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)

    # get one batch of given size
    def get_batch(self) -> list:
        return random.sample(self.memory, self.batch_size)

    # update epsilon
    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    # moving with pole
    def move(self, env, state):
        if random.random() < self.epsilon:
            # Exploration
            return env.action_space.sample()
        else:
            # Exploitation
            q_values = self.model.predict(self.preprocess_input(state))[0]

            return np.argmax(q_values)

    # reshaping input to DQN
    def preprocess_input(self, state):
        return np.reshape(state, (1, 4))

    # update Q-values from batch
    def update(self):
        mini_batch = self.get_batch()
        new_q_values = []

        for state, action, reward, next_state, done in mini_batch:
            # calculating Q(s,a)
            q_values_state = self.model.predict(self.preprocess_input(state))[0]

            if done:
                target = reward
            else:
                # choosing max action from state_
                max_action = np.argmax(self.model.predict(self.preprocess_input(next_state)))
                # calculating target and Q(s_,a)
                target = reward + self.gamma * self.target_model.predict(self.preprocess_input(next_state))[0][
                    max_action]

            # updated Q values for specific action
            q_values_state[action] = target
            new_q_values.append(q_values_state)

        input_states = np.array([sample[0] for sample in mini_batch])
        self.model.fit(input_states, np.array(new_q_values), verbose=0, epochs=1, batch_size=self.batch_size)

    # copy weights from model to target
    def update_target_weights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for index, (weight, target_weight) in enumerate(zip(weights, target_weights)):
            target_weight = weight * self.tau + target_weight * (1 - self.tau)
            target_weights[index] = target_weight

        self.target_model.set_weights(target_weights)
