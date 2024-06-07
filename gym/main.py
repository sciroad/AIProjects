import random

import numpy as np
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import gym

env = gym.make('CartPole-v1')

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential([
    Flatten(input_shape=(1, states)),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(actions, activation='linear')

])

agent = DQNAgent(model=model, memory=SequentialMemory(limit=50000, window_length=1),
                 policy=BoltzmannQPolicy(), nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2,
                 )

agent.compile(Adam(lr=1e-3), metrics=['mae'])

agent.fit(env, nb_steps=50000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=100, visualize=False)
print('Average score over 100 test games:{}'.format(
    np.mean(results.history['episode_reward'])))

env.close()

episodes = 10

# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = random.choice([0, 1])
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))

# env.close()
