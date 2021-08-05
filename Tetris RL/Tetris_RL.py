
import tensorflow as tf
import time
import numpy as np
import random
import sys
import ctypes
from collections import deque
import matplotlib.pyplot as plt
import os

import tetrisboard

EXPERIENCE_BUFFER_SIZE = 10000
EXPERIENCE_SAMPLE_SIZE = 16

EPSILON_DECAY_RATE = .9999
EPSILON_MIN = .02
SYNC_FREQUENCY = 400
TRAIN_FREQUENCY = 64
LEARNING_RATE = 1e-3

DISCOUNT_RATE = .99

action_map = [
    "rleft",
    "rright",
    "left",
    "right",
    "down"
]


class MemoryFragment:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class Memory:
    def __init__(self):
        #self.memory = SumTree()
        self.memory = deque([], 10000)

    def addToMemory(self, state, action, reward, next_state):
        fragment = MemoryFragment(state, action, reward, next_state)
        #self.memory.insert(fragment)
        self.memory.append(fragment)

    #def discountRewards(self, action_count, discount_rate = .99):
    #    discounted_rewards = np.zeros_like(self.rewards)
    #    R = 0
    #    for t in reversed(range(len(self.rewards) - action_count, len(self.rewards))):
    #        R = R * discount_rate + self.rewards[t]
    #        self.rewards[t] = R

    def sampleMemory(self, batch_size=EXPERIENCE_SAMPLE_SIZE):
        if len(self.memory) > EXPERIENCE_BUFFER_SIZE:
            sample_indices = random.sample(range(len(self.memory)), EXPERIENCE_SAMPLE_SIZE)
        else:
            sample_indices = range(len(self.memory))

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in range(len(sample_indices)):
            states.append(self.memory[sample_indices[i]].state)
            actions.append(self.memory[sample_indices[i]].action)
            rewards.append(self.memory[sample_indices[i]].reward)
            next_states.append(self.memory[sample_indices[i]].next_state)

        return (states, actions, rewards, next_states)

    #def updateLogits(self, model):
    #    self.memory.updateLogits(model)

    #def updateTargetLogits(self, target):
    #    self.memory.updateTargetLogits(target)


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(200),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation=None),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(1e-4), loss='mse')

    return model


def getStateActionPredictions(model, observation):

    logit_list = model.predict(np.expand_dims(observation, 0))[0][0]

    return logit_list



def train_step(model, target, memory):

    states, actions, rewards, next_states = memory.sampleMemory()

    logits = model.predict(np.array(states))

    for i in range(len(states)):

        q = rewards[i]

        if not next_states[i] is None:
            q += np.max(target.predict(np.expand_dims(next_states[i], 0))) * DISCOUNT_RATE

        logits[i][actions[i]] = (1 - LEARNING_RATE) * logits[i][actions[i]] + LEARNING_RATE * q

    model.fit(np.array(states), logits, batch_size = len(states), epochs = 3, verbose=0)

def syncModels(model, target):
    for layer_i in range(len(target.layers)):
        if len(target.layers[layer_i].get_weights()) != 0:
            weights_shape = target.layers[layer_i].get_weights()[0].shape
            biases_shape = target.layers[layer_i].get_weights()[1].shape

            weights = np.copy(target.layers[layer_i].get_weights()[0]).flatten()
            new_weights = np.copy(model.layers[layer_i].get_weights()[0]).flatten()
            for i in range(weights.shape[0]):
                weights[i] = new_weights[i]

            biases = np.copy(target.layers[layer_i].get_weights()[1]).flatten()
            new_biases = np.copy(model.layers[layer_i].get_weights()[1]).flatten()
            for i in range(biases.shape[0]):
                biases[i] = new_biases[i]

            weights = np.reshape(weights, weights_shape)
            biases = np.reshape(biases, biases_shape)
            
            target.layers[layer_i].set_weights([weights, biases])

if __name__ == "__main__":
    board = tetrisboard.tetrisboard()

    model = createModel()
    target_model = createModel()
    action = np.argmax(getStateActionPredictions(model, board.getObservation()))
    
    if os.path.isdir('./model'):
        print("Loading model...")
        model = tf.keras.models.load_model('./model')
        target_model = tf.keras.models.load_model('./model')
        print("Model loaded")

    memory = Memory()

    epsilon = 1.
    sync_counter = 0
    max_score = 0
    step = 1

    if os.path.isfile('./vars.txt'):
        f = open('vars.txt', 'r')
        data = f.read().split('\n')
        epsilon = float(data[0])
        step = int(data[1])
        max_score = float(data[2])

        f.close()

    for i_episode in range(step, 5000):
        state = board.getObservation()

        while True:

            if random.random() < epsilon:
                action = random.randint(0, 4)
                #print("LOGIT: ", logit)
            else:
                action = np.argmax(getStateActionPredictions(model, state))

            epsilon = max(epsilon * EPSILON_DECAY_RATE, EPSILON_MIN)

            old_score = board.score

            continue_flag = board.step(action_map[action])
            reward = board.score - old_score

            new_state = board.getObservation()

            if not continue_flag:
                reward -= 100.
                new_state = None

            memory.addToMemory(state, action, reward, new_state)

            if not continue_flag:
                print("Score: ", board.score)

                if board.score > max_score:
                    max_score = board.score

                print("Highscore: ", max_score)

                print("Epsilon: ", epsilon)

                board.printBoard()

                board.clear()

                break

            if (step % 4) == 0:
                train_step(model, target_model, memory)

            if step == SYNC_FREQUENCY:
                sync_counter = 0
                syncModels(model, target_model)
                model.save('./model')
                f = open('./vars.txt', 'w')
                f.write(str(epsilon) + '\n' + str(i_episode) + '\n' + str(max_score))
                f.close()

                break

            step += 1

            state = np.copy(new_state)
