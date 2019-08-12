import csv
import os
from matplotlib import pyplot as plt
import gym
import numpy as np

from DQN import DQN

TEST_INDEX = 1
MAX_EPISODES = 300
MAX_TIME_STEPS = 200


# writing parameters of DQN to CSV file
def write_results(agent: DQN, episode: int):
    file_path = 'results/results.csv'

    # append header for the first time
    if not os.path.isfile(file_path):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["EPISODES", "ALPHA", "EPSILON", "GAMMA", "TAU", "BATCH_SIZE",
                 "BUFFER_SIZE", "MIN_MEMORY_SIZE"])

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [episode,
             agent.learning_rate,
             agent.epsilon,
             agent.gamma,
             agent.tau,
             agent.batch_size,
             agent.buffer_size,
             agent.min_memory_size])


# plotting number of steps in episodes
def plot_episodes(episodes_length: list):
    plt.plot(episodes_length)
    plt.ylabel('Steps')
    plt.xlabel('Episode')
    plt.title('Double DQN CartPole v-0')
    plt.savefig(f'graphs/test_{TEST_INDEX}.png')


class Main:
    env = gym.make('CartPole-v0')
    env.seed(1)
    episodes_scores = []
    episodes_length = []
    current_episode = 0
    agent = DQN(learning_rate=0.001,
                epsilon=0.5,
                epsilon_decay=0.99,
                gamma=0.99,
                batch_size=64,
                buffer_size=2000,
                min_memory_size=500,
                tau=0.1)
    agent.model = agent.create_model()
    agent.target_model = agent.create_model()

    for current_episode in range(MAX_EPISODES):
        state = env.reset()
        cum_reward = 0

        for time_step in range(MAX_TIME_STEPS):
            # env.render()
            action = agent.move(env, state)
            state_, reward, done, info = env.step(action)
            cum_reward += reward
            agent.store_experience([state, action, reward, state_, done])
            state = state_
            if len(agent.memory) > agent.min_memory_size:
                agent.update()
                agent.update_target_weights()

            if done:
                episodes_length.append(time_step)
                break

        agent.update_epsilon()
        episodes_scores.append(cum_reward)
        last_100_avg = np.mean(episodes_scores[-100:])
        print(f'Episode {current_episode} with length {episodes_length[-1:]} average score was {last_100_avg}')

        if len(episodes_scores) >= 100 and last_100_avg >= 195.0:
            print(f'Solved in {current_episode} episodes')
            break
    write_results(agent, current_episode)
    plot_episodes(episodes_length)
    env.close()
