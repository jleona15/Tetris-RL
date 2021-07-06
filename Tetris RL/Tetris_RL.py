
import tensorflow as tf
import time
import numpy as np
from PIL import ImageGrab, Image
import pydirectinput
import os
import random
import msvcrt
from collections import deque

BLOCK_Z = 0
BLOCK_L = 1
BLOCK_O = 2
BLOCK_S = 3
BLOCK_I = 4
BLOCK_J = 5
BLOCK_T = 6

BOARD_EMPTY = 0.
BOARD_OCCUPIED = 1.
BOARD_ACTIVE = 2.

action_map = [
    "rleft",
    "rright",
    "left",
    "right",
    "down"
]

class TetrisSimulation:
    def __init__(self):
        self.clear()

    def clear(self):
        self.score = 0
        self.board = np.zeros((10, 20), dtype=np.uint8)
        self.active_indices = []
        self.active_type = 0
        self.rotation_state = 0
        self.generateNewPiece()
        self.ticks_since_down = 0

    def generateNewPiece(self, new_block=-1):
        if new_block == -1:
            new_block = random.randint(BLOCK_Z, BLOCK_T)

        new_active_indices = []

        if new_block == BLOCK_Z:
            new_active_indices = [(4, 0), (5, 0), (5, 1), (6, 1)]
        elif new_block == BLOCK_L:
            new_active_indices = [(4, 0), (4, 1), (5, 0), (6, 0)]
        elif new_block == BLOCK_O:
            new_active_indices = [(4, 0), (4, 1), (5, 0), (5, 1)]
        elif new_block == BLOCK_S:
            new_active_indices = [(4, 1), (5, 0), (5, 1), (6, 0)]
        elif new_block == BLOCK_I:
            new_active_indices = [(3, 0), (4, 0), (5, 0), (6, 0)]
        elif new_block == BLOCK_J:
            new_active_indices = [(4, 0), (5, 0), (6, 0), (6, 1)]
        elif new_block == BLOCK_T:
            new_active_indices = [(4, 0), (5, 0), (5, 1), (6, 0)]

        for i in new_active_indices:
            if self.board[i[0]][i[1]] == BOARD_OCCUPIED:
                return False
            else:
                self.board[i[0]][i[1]] = BOARD_ACTIVE

        for i in self.active_indices:
            if(i[1] < 0):
                return False
            else:
                self.board[i[0]][i[1]] = BOARD_OCCUPIED
                
        self.active_indices = new_active_indices
        self.active_type = new_block
        self.rotation_state = 0

        return True

    def rotateLeft(self):
        self.ticks_since_down += 1

        new_pos = []

        if self.active_type == BLOCK_O:
            return True
        elif self.active_type == BLOCK_Z:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[1]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y + 1)]

        elif self.active_type == BLOCK_L:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x += 1
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y + 2)]

            elif self.rotation_state == 3:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y - 1),
                           (top_left_x + 2, top_left_y)]

            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 2)]

            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y)]

        elif self.active_type == BLOCK_S:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[1]
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 2)]
            else:
                (top_left_x, top_left_y) = self.active_indices[1]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y - 1)]
        elif self.active_type == BLOCK_I:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[2]
                top_left_y -= 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x, top_left_y + 3)]
            else:
                (top_left_x, top_left_y) = self.active_indices[2]
                top_left_x -= 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y),
                           (top_left_x + 3, top_left_y)]
        elif self.active_type == BLOCK_J:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x += 1
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y)]
                
            elif self.rotation_state == 3:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y + 1)]
            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y += 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 2),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y),
                           (top_left_x + 2, top_left_y + 1)]
        elif self.active_type == BLOCK_T:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x += 1
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y + 1)]
            elif self.rotation_state == 3:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y)]
            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y)]

        for i in new_pos:
            if i[0] < 0 or i[0] > 9 or i[1] > 19 or (i[1] > 0 and self.board[i[0]][i[1]] == BOARD_OCCUPIED):
                return False

        for i in range(4):
            if self.active_indices[i][1] >= 0:
                self.board[self.active_indices[i][0]][self.active_indices[i][1]] = BOARD_EMPTY
            
        for i in range(4):    
            if new_pos[i][1] >= 0:
                self.board[new_pos[i][0]][new_pos[i][1]] = BOARD_ACTIVE

        self.active_indices = new_pos            

        if self.active_type == BLOCK_S or self.active_type == BLOCK_Z or self.active_type == BLOCK_I:
            self.rotation_state = 1 - self.rotation_state
        else:
            if self.rotation_state > 0:
                self.rotation_state -= 1
            else:
                self.rotation_state = 3



        return True

    def rotateRight(self):
        self.ticks_since_down += 1

        new_pos = []

        if self.active_type == BLOCK_O:
            return True
        elif self.active_type == BLOCK_Z:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[1]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y + 1)]

        elif self.active_type == BLOCK_L:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 2)]

            elif self.rotation_state == 1:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y - 1),
                           (top_left_x + 2, top_left_y)]

            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y -= 1
                top_left_x += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y + 2)]

            else:
                (top_left_x, top_left_y) = self.active_indices[1]
                top_left_x -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y)]

        elif self.active_type == BLOCK_S:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[1]
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 2)]
            else:
                (top_left_x, top_left_y) = self.active_indices[1]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y - 1)]
        elif self.active_type == BLOCK_I:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[2]
                top_left_y -= 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x, top_left_y + 3)]
            else:
                (top_left_x, top_left_y) = self.active_indices[2]
                top_left_x -= 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y),
                           (top_left_x + 3, top_left_y)]
        elif self.active_type == BLOCK_J:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 2),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y)]
            elif self.rotation_state == 1:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_y -= 2
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y + 1)]
            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y),
                           (top_left_x + 2, top_left_y + 1)]
        elif self.active_type == BLOCK_T:
            if self.rotation_state == 0:
                (top_left_x, top_left_y) = self.active_indices[0]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1)]
            elif self.rotation_state == 1:
                (top_left_x, top_left_y) = self.active_indices[0]
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y - 1),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 2, top_left_y)]
            elif self.rotation_state == 2:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x += 1
                top_left_y -= 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x, top_left_y + 1),
                           (top_left_x, top_left_y + 2),
                           (top_left_x + 1, top_left_y + 1)]
            else:
                (top_left_x, top_left_y) = self.active_indices[0]
                top_left_x -= 1
                top_left_y += 1
                new_pos = [(top_left_x, top_left_y),
                           (top_left_x + 1, top_left_y),
                           (top_left_x + 1, top_left_y + 1),
                           (top_left_x + 2, top_left_y)]

        for i in new_pos:
            if i[0] < 0 or i[0] > 9 or i[1] > 19 or (i[1] > 0 and self.board[i[0]][i[1]] == BOARD_OCCUPIED):
                return False

        for i in range(4):
            if self.active_indices[i][1] >= 0:
                self.board[self.active_indices[i][0]][self.active_indices[i][1]] = BOARD_EMPTY
            
        for i in range(4):    
            if new_pos[i][1] >= 0:
                self.board[new_pos[i][0]][new_pos[i][1]] = BOARD_ACTIVE

        self.active_indices = new_pos            

        if self.active_type == BLOCK_S or self.active_type == BLOCK_Z or self.active_type == BLOCK_I:
            self.rotation_state = 1 - self.rotation_state
        else:
            if self.rotation_state < 3:
                self.rotation_state += 1
            else:
                self.rotation_state = 0

        return True

    def printBoard(self):
        print('==========')
        for j in range(20):
            for i in range(10):
                if(self.board[i][j] == BOARD_EMPTY):
                    print(' ', end='')
                elif(self.board[i][j] == BOARD_OCCUPIED):
                    print('X', end='')
                else:
                    print('*', end='')
            print()
        print('==========')

    def moveLeft(self):
        self.ticks_since_down += 1

        for i in range(4):
            if self.active_indices[i][0] - 1 < 0 or (self.active_indices[i][1] >= 0 and self.board[self.active_indices[i][0] - 1][self.active_indices[i][1]] == BOARD_OCCUPIED):
                return False

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_EMPTY

        for i in range(4):
            self.active_indices[i] = (self.active_indices[i][0] - 1, self.active_indices[i][1])

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_ACTIVE

        return True

    def moveRight(self):
        self.ticks_since_down += 1

        for i in range(4):
            if self.active_indices[i][0] + 1 > 9 or (self.active_indices[i][1] >= 0 and self.board[self.active_indices[i][0] + 1][self.active_indices[i][1]] == BOARD_OCCUPIED):
                return False

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_EMPTY

        for i in range(4):
            self.active_indices[i] = (self.active_indices[i][0] + 1, self.active_indices[i][1])

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_ACTIVE

        return True

    def moveDown(self):
        self.ticks_since_down = 0

        for i in range(4):
            if self.active_indices[i][1] + 1 > 19 or (self.active_indices[i][1] >= 0 and self.board[self.active_indices[i][0]][self.active_indices[i][1] + 1] == BOARD_OCCUPIED):
                return False

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_EMPTY

        for i in range(4):
            self.active_indices[i] = (self.active_indices[i][0], self.active_indices[i][1] + 1)

        for i in self.active_indices:
            if i[1] >= 0:
                self.board[i[0]][i[1]] = BOARD_ACTIVE

        return True

    def clearLines(self):
        clear_count = 0

        for j in range(20):
            line_full = True
            for i in range(10):
                if self.board[i][j] != BOARD_OCCUPIED:
                    line_full = False
                    break

            if line_full:
                clear_count += 1
                for j2 in range(j, 0, -1):
                    for i in range(10):
                        if self.board[i][j2 - 1] != BOARD_ACTIVE:
                            self.board[i][j2] = self.board[i][j2 - 1]
                for i in range(10):
                    if self.board[i][0] == BOARD_OCCUPIED:
                        self.board[i][0] = BOARD_EMPTY

        if clear_count == 1:
            self.score += 40
        elif clear_count == 2:
            self.score += 100
        elif clear_count == 3:
            self.score += 300
        elif clear_count == 4:
            self.score += 1200
        else:
            self.score += 2

    def dbgPoints(self):
        score_to_add = 0
        for i in self.active_indices:
            if i[0] <= 4:
                score_to_add = max(score_to_add, 5 - i[0])
            else:
                score_to_add = max(score_to_add, 4 - i[0])
        self.score += score_to_add

    def step(self, action):
        ret = True

        if action == "rleft":
            self.rotateLeft()
        elif action == "rright":
            self.rotateRight()
        elif action == "left":
            self.moveLeft()
        elif action == "right":
            self.moveRight()
        else:
            if not self.moveDown():
                #self.dbgPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        if self.ticks_since_down >= 8:
            if not self.moveDown():
                #self.dbgPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        return ret


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = deque(maxlen=10000)
        self.actions = deque(maxlen=10000)
        self.rewards = deque(maxlen=10000)

    def addToMemory(self, new_observation, new_action, new_reward):
        self.states.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def discountRewards(self, action_count, discount_rate = .99):
        discounted_rewards = np.zeros_like(self.rewards)
        R = 0
        for t in reversed(range(len(self.rewards) - action_count, len(self.rewards))):
            R = R * discount_rate + self.rewards[t]
            self.rewards[t] = R

    def sampleMemory(self, batch_size=256):
        indices = np.random.choice(len(self.states), batch_size, False)

        observations = []
        actions = []
        rewards = []

        for i in indices:
            observations.append(self.states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])

        return np.array(observations), np.array(actions), np.array(rewards)


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 5, 1, activation='relu', input_shape=(10, 20, 1)),
        tf.keras.layers.Conv2D(64, (1,16), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation=None)
    ])

    return model


def nextAction(model, observation):
    logits = model.predict(np.expand_dims(observation, (0, -1)))

    return logits.argmax()


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)

    return loss

def train_step(model, target, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        logits = target.__call__(np.expand_dims(observations, -1))

        loss = compute_loss(logits, actions, rewards)

    grads = tape.gradient(loss, target.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def syncModels(model, target, ratio):
    for layer_i in range(len(target.layers)):
        if len(target.layers[layer_i].get_weights()) != 0:
            weights_shape = target.layers[layer_i].get_weights()[0].shape
            biases_shape = target.layers[layer_i].get_weights()[1].shape

            weights = np.copy(target.layers[layer_i].get_weights()[0]).flatten()
            new_weights = np.copy(model.layers[layer_i].get_weights()[0]).flatten()
            for i in random.sample(range(weights.shape[0]), k = int(weights.shape[0] * ratio)):
                weights[i] = new_weights[i]

            biases = np.copy(target.layers[layer_i].get_weights()[1]).flatten()
            new_biases = np.copy(model.layers[layer_i].get_weights()[1]).flatten()
            for i in random.sample(range(biases.shape[0]), k = int(biases.shape[0] * ratio)):
                biases[i] = new_biases[i]

            weights = np.reshape(weights, weights_shape)
            biases = np.reshape(biases, biases_shape)
            
            target.layers[layer_i].set_weights([weights, biases])

if __name__ == "__main__":

    board = TetrisSimulation()

    model = createModel()
    target_model = createModel()
    memory = Memory()

    learning_rate = 1e-4
    epsilon = 1.
    epsilon_decay_rate = .99999
    epsilon_min = .02
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    sync_frequency = 10
    sync_rate = 1.0

    sync_counter = 0

    max_score = 0

    for i_episode in range(5000):
        observation = np.copy(board.board)
        print("Step: ", i_episode)

        action_counter = 0

        while True:
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = nextAction(model, observation)
            
            action_counter += 1

            epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

            old_score = board.score

            continue_flag = board.step(action_map[action])
            reward = (board.score - old_score) * 1.0

            if not continue_flag:
                reward -= 100

            new_observation = np.copy(board.board)

            memory.addToMemory(observation, action, reward)

            if not continue_flag:
                memory.discountRewards(action_counter)
                observations, actions, discounted_rewards = memory.sampleMemory()
                train_step(model, target_model, optimizer, observations, actions, discounted_rewards)

                #print("Step ", i_episode, " ended")
                print("Score: ", board.score)

                if board.score > max_score:
                    max_score = board.score

                print("Highscore: ", max_score)

                board.printBoard()

                board.clear()
                break

                sync_counter += 1

                if sync_counter == sync_frequency:
                    sync_counter = 0
                    syncModels(model, target_model, sync_ratio)


            observation = new_observation

    model.save('model')