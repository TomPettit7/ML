import gym
import numpy as np

env = gym.make('Roulette-v0')

learning_rate = 0.8
gamma = 0.8

reward_list = []
num_episodes = 150000
print(env.observation_space)
print(env.action_space)
Q = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(num_episodes):
    total_score = 0
    j = 0

    observation = env.reset()

    while j < 99:
        j += 1

        action = np.argmax(Q[observation, :] + (np.random.randn(1, env.action_space.n)*(1.0/(i+1))) )

        new_observation, reward, done, _ = env.step(action)

        total_score += reward

        Q[observation, action] = Q[observation, action] + learning_rate*(reward + gamma*np.max(Q[new_observation, :] - Q[observation, action]))

        total_score += reward

        observation = new_observation

        if done:
            break

    reward_list.append(total_score)

print('Final Score: '+str(sum(reward_list) / num_episodes))
