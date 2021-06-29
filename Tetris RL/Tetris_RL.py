
import tensorflow as tf
import datetime
import numpy as np
from PIL import ImageGrab
from PIL import Image

masks = [
    Image.open("mask0.bmp").convert('L').getdata(),
    Image.open("mask1.bmp").convert('L').getdata(),
    Image.open("mask2.bmp").convert('L').getdata(),
    Image.open("mask3.bmp").convert('L').getdata(),
    Image.open("mask4.bmp").convert('L').getdata(),
    Image.open("mask5.bmp").convert('L').getdata(),
    Image.open("mask6.bmp").convert('L').getdata(),
    Image.open("mask7.bmp").convert('L').getdata(),
    Image.open("mask8.bmp").convert('L').getdata(),
    Image.open("mask9.bmp").convert('L').getdata()
]

currentScore = 0

def clarifyDigit(im):
    im = im.convert('L').getdata()
    min_i = 0
    diff = 0

    for i in range(len(im)):
        diff += abs(im[i] - masks[0][i])

    for i in range(1, 10):
        tmp_diff = 0
        for j in range(len(im)):
            tmp_diff += abs(im[j] - masks[i][j])

        if diff > tmp_diff:
            min_i = i
            diff = tmp_diff

    return min_i

def getScore():
    initial_x = 235
    initial_y = 119

    x_width = 8
    y_width = 7

    x_step = x_width + 1

    score = 0

    im = ImageGrab.grab((initial_x, initial_y, initial_x + 6 * x_width + 5, initial_y + y_width))
    #im.show()

    #Pull the score digits from the screen
    for i in range(6):
        bounding_box = (initial_x + i * x_step, initial_y, initial_x + i * x_step + x_width, initial_y + y_width)
    #    score = score * 10 + clarifyDigit(ImageGrab.grab(bounding_box))
        score = score * 10 + clarifyDigit(im.crop(bounding_box))

    return score


def getBoard():
    initial_x = 126
    initial_y = 103

    x_step = 9
    y_step = 8

    board = np.ndarray((10, 20), dtype=np.float)

    im = ImageGrab.grab((initial_x, initial_y, initial_x + 10 * x_step, initial_y + 20 * y_step))

    initial_x = (x_step / 2)
    initial_y = (y_step / 2)

    for i in range(10):
        for j in range(20):
            p = im.getpixel((initial_x + x_step * i, initial_y + y_step * j))
            if p[0] + p[1] + p[2] > 0:
                board[i][j] = 1
            else:
                board[i][j] = 0

    return board


def getInput():
    score = getScore()
    board = getBoard()

    return (score, board)


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def addToMemory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

def aggregate_memories(memories):
    batch_memory = Memory()

    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.addToMemory(*step)

    return batch_memory


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    return model


def nextAction(model, observation):
    logits = model.predict(observation)

    return logits.argmax()


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


def discount_rewards(rewards, discount_rate = 0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * discount_rate + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)

    return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)

        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == "__main__":
    model = createModel()
    memory = Memory()

    getInput()