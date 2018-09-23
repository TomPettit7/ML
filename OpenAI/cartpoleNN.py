import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf


def create_data(env):

    scores = []
    score_requirement = 50
    X_data = []
    y_data = []

    for i in range(10000):
        observation = env.reset()
        score = 0
        move_memory = []
        obs_memory = []

        for i in range(500):
            action = np.random.randint(0,2)
            one_hot = np.zeros(2)

            one_hot[action] = 1

            move_memory.append(one_hot)
            obs_memory.append(observation)

            observation, reward, done, _ = env.step(action)
            score += reward

            if done:
                break

        if score >= score_requirement:
            scores.append(score)

            X_data += obs_memory
            y_data += move_memory

    print('Mean Score of Training Data: ' + str(np.mean(scores)))
    print('Number of Training Data Points: ' + str(len(scores)))

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def train_NN_model(X_data, y_data):

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_shape=(4,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_data, y_data, epochs=5)

    return model


def predict():
    scores = []
    env = gym.make('CartPole-v0')
    X_data, y_data = create_data(env)
    sim_steps = 500

    model = train_NN_model(X_data, y_data)

    for i in range(100):
        observation = env.reset()
        score = 0

        for step in range(sim_steps):
            action = np.argmax(model.predict(observation.reshape(1,4)))

            observation, reward, done, _ = env.step(action)

            score += reward

            if done:
                break

        scores.append(score)

    print('Average Score After Testing: ' + str(np.mean(scores)))

predict()

#Average score of 200 over 100 consecutive trials

#Therefore SOLVED
