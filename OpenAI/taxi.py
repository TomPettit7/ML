import gym
import numpy as np

env = gym.make('Taxi-v2')

#create the Q Table
Q = np.zeros([env.observation_space.n, env.action_space.n])

#set paramaters
learning_rate = 0.8
gamma = 0.95

reward_list = []

#start training
for i in range(7500):

    #initialise variables
    observation = env.reset()
    total_reward = 0
    done = False
    j = 0

    # Each game has 99 actions
    while j < 99:
        j += 1
        
        # Take the best action given the state from the Q Table
        # Add this to a random action and multiply this random action by a fraction
        # This fraction is programmed to decrease every iteration in order to show how the Q Table is becoming better over time
        # This covers for missing values in the Q Table, such as when it first runs
        
        action = np.argmax(Q[observation, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i+1)))
        
        # Retrieve information about the action taken
        new_observation, reward, done, _ = env.step(action)
        
        # The Q Learning Algorithm using the Bellman Equation
        Q[observation, action] = Q[observation, action] + learning_rate*(reward + gamma*np.max(Q[new_observation, :]) - Q[observation, action])
        
        # Add reward to total reward so that score can be calculated
        total_reward += reward
        
        # Set the variable observation as the new observation, as otherwise it would take the first observation value from when the bot first runs
        observation = new_observation

        if done == True:
            break

    reward_list.append(total_reward)

print('Score over time: ' + str(sum(reward_list) / 2000))
