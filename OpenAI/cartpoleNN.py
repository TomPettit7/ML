import gym
import random
import numpy as np 
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median 
from collections import Counter


env = gym.make('CartPole-v0')
env.reset()
required_score = 50
prev_obs = []
output = [0,0]

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation = 'softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model 

def create_data():
    scores = []
    #observations, moves
    training_data = []
    accepted_scores = []

    for _ in range(10000):
        global prev_obs
        score = 0
        #previous observations, action fulfilled
        game_memory = []
        prev_obs = []

        for _ in range(500):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])
        
            prev_obs = observation
            score += reward 

        if score >= required_score:
            accepted_scores.append(score)
            for data in game_memory:
                global output
                if data[1] == 1:
                    output = [0,1]
                if data[1] == 0:
                    output = [1,0]
                
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    print('Average Score: ', (sum(scores)/len(scores)))
    return training_data

def train_model(training_data, model=False):
    global prev_obs
    global output
    #X: prev_obs
    #Y: output
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)

    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))
    model.fit({'input': X}, {'targets': Y}, n_epoch = 3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = create_data()
model = train_model(training_data)

scores = []
choices =[]
for iter in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(500):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([prev_obs, action])
        score += reward
        if done:
            break

    scores.append(score)

print('New Average Score: ', sum(scores)/len(scores))
print('Choice: Right: {}, Choice: Left: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
