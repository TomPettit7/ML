import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
learning_rate = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
rewardList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    total_reward_for_this_episode = 0
    done = False
    j=0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        #Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)
        #Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + learning_rate*(reward + y*np.max(Q[new_state, :]) - Q[state, action])
        total_reward_for_this_episode += reward
        state = new_state
        if done == True:
            break
    print(total_reward_for_this_episode)
    rewardList.append(total_reward_for_this_episode)

print("Score over time: " + str(sum(rewardList) / num_episodes))
print("Final Q-Table Values")
print(Q)
