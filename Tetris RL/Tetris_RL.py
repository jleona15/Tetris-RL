
import tensorflow as tf
import time
import numpy as np
import os
import random
import sys
import ctypes
import msvcrt
from collections import deque

BLOCK_Z = 0
BLOCK_L = 1
BLOCK_O = 2
BLOCK_S = 3
BLOCK_I = 4
BLOCK_J = 5
BLOCK_T = 6

BOARD_EMPTY = -1.
BOARD_OCCUPIED = 1.
BOARD_ACTIVE = 0.

EXPERIENCE_BUFFER_SIZE = 10000
EXPERIENCE_SAMPLE_SIZE = 64

DISCOUNT_RATE = .99

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
        self.score = 0.
        self.board = np.ndarray((10, 20), dtype=np.float32)

        for i in range(10):
            for j in range(20):
                self.board[i][j] = BOARD_EMPTY

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

    def rowCountPoints(self):
        rowCounts = np.zeros(20, dtype=np.uint8)

        for i in range(10):
            for j in range(20):
                if self.board[i][j] == BOARD_OCCUPIED:
                    rowCounts[j] += 1

        for i in self.active_indices:
            if rowCounts[i[1]] == 6:
                self.score += 1
            elif rowCounts[i[1]] == 7:
                self.score += 3
            elif rowCounts[i[1]] == 8:
                self.score += 7

            rowCounts[i[1]] += 1

    def dbgPoints(self):
        score_to_add = 0
        for i in self.active_indices:
            if i[0] <= 4:
                score_to_add += 5 - i[0]
            else:
                score_to_add += 4 - i[0]
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
                #self.rowCountPoints()
                self.dbgPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        if self.ticks_since_down >= 8:
            if not self.moveDown():
                #self.rowCountPoints()
                self.dbgPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        self.score -= .01

        return ret

class SumTree:
    def __init__(self, max_depth = 12):
        self.tree = np.zeros((2**max_depth) - 1, dtype = float)
        self.arr = np.ndarray(2**max_depth, dtype = MemoryFragment)
        self.next_insert_index = 0
        self.max_depth = max_depth

    def SUM_CHECK(self):
        depth = 0
        index = 0
        while index * 2 + 2 < self.tree.shape[0]:
            if index + 1 == 2**(depth + 1):
                depth += 1
            if abs(self.tree[index] - (self.tree[index * 2 + 1] + self.tree[index * 2 + 2])) > .00001:
                print("Sum Check error at tree index ", index, ", depth ", depth)
                print(self.tree[index], " != ", self.tree[index * 2 + 1], ' + ', self.tree[index * 2 + 2], ' = ',  self.tree[index * 2 + 1] + self.tree[index * 2 + 2])
                1 / 0

            index += 1

        depth += 1

        while index < self.tree.shape[0]:
            sum_l = 0.
            sum_r = 0.

            if not self.arr[(index * 2 + 1) - self.tree.shape[0]] is None:
                sum_l = self.arr[(index * 2 + 1) - self.tree.shape[0]].loss()

            if not self.arr[(index * 2 + 2) - self.tree.shape[0]] is None:
                sum_r = self.arr[(index * 2 + 2) - self.tree.shape[0]].loss()

            if abs(self.tree[index] - (sum_l + sum_r)) > .00001:
                print("Sum check error at tree index ", index, ", depth ", depth)
                print(self.tree[index], " != ", sum_l, ' + ', sum_r, ' = ',  sum_l + sum_r)
                1 / 0

            index += 1

    def sample(self, batch_size):
        s = []
        #self.SUM_CHECK();
        for i in range(batch_size):
            s.append(random.random() * self.tree[0])

        return self.sampleHelper(s, 0)

    def sampleHelper(self, s, i):
        if s == []:
            return []
        if i * 2 + 2 < self.tree.shape[0]:

            left_s = []
            right_s = []

            for j in s:
                if self.tree[i * 2 + 1] >= j:
                    left_s.append(j)
                else:
                    right_s.append(j - self.tree[i * 2 + 1])

            #print("##############################")
            #print(s)
            #print(self.tree[i * 2 + 1])
            #print(left_s)
            #print(right_s)
            #print("##############################")

            #print("Left sum: ", self.left.sum)
            #print("Right sum: ", self.right.sum)

            #print("Left samples: ", left_s)
            #print("Right samples: ", right_s)

            return self.sampleHelper(left_s, i * 2 + 1) + self.sampleHelper(right_s, i * 2 + 2)
        
        #print("Parent index: ", (i - 1) // 2)
        #print(self.tree[(i - 1) // 2])

        ret = []
        l_i = i * 2 + 1 - self.tree.shape[0]
        r_i = i * 2 + 2 - self.tree.shape[0]

        #print(i)
        #print(self.arr[0])

        for j in s:
            if self.arr[l_i].loss() >= j:
                ret.append(self.arr[l_i])
            else:
                ret.append(self.arr[r_i])

        return ret

    def insert(self, fragment):
        if self.next_insert_index < self.arr.shape[0]:
            self.arr[self.next_insert_index] = fragment;
            index = self.next_insert_index
            self.next_insert_index += 1
            loss_delta = fragment.loss()
        else:
            index = 0
            sample = random.random() * self.tree[0]

            while index * 2 + 2 < self.tree.shape[0]:
                if self.tree[index * 2 + 1] < sample:
                    index = index * 2 + 1
                else:
                    sample -= self.tree[index * 2 + 1]
                    index = index * 2 + 2

            if self.arr[index * 2 + 1 - self.tree.shape[0]].loss() < sample:
                loss_delta = fragment.loss() - self.arr[index * 2 + 1 - self.tree.shape[0]].loss()
                self.arr[index * 2 + 1 - self.tree.shape[0]] = fragment
            else:
                loss_delta = fragment.loss() - self.arr[index * 2 + 2 - self.tree.shape[0]].loss()
                self.arr[index * 2 + 2 - self.tree.shape[0]] = fragment

        index = ((index + self.tree.shape[0]) - 1) // 2

        while index >= 0:
            self.tree[index] += loss_delta
            index = (index - 1) // 2

        #self.SUM_CHECK()

    def updateLogits(self, model):
        for i in range(self.arr.shape[0]):
            if not self.arr[i] is None:
                self.arr[i].updateLogits(model)

        for i in range(2**(self.max_depth - 1) - 1, self.tree.shape[0]):
            new_sum = 0.

            if not self.arr[(i * 2 + 1) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 1) - (self.tree.shape[0])].loss()

            if not self.arr[(i * 2 + 2) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 2) - (self.tree.shape[0])].loss()

            self.tree[i] = new_sum

        for i in range(2**(self.max_depth - 1) - 2, -1, -1):
            self.tree[i] = self.tree[i * 2 + 1] + self.tree[i * 2 + 2]



    def updateTargetLogits(self, target_model):
        for i in range(self.arr.shape[0]):
            if not self.arr[i] is None:
                self.arr[i].updateTargetLogits(model)

        for i in range(2**(self.max_depth - 1) - 1, self.tree.shape[0]):
            new_sum = 0.

            if not self.arr[(i * 2 + 1) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 1) - (self.tree.shape[0])].loss()

            if not self.arr[(i * 2 + 2) - (self.tree.shape[0])] is None:
                new_sum += self.arr[(i * 2 + 2) - (self.tree.shape[0])].loss()

            self.tree[i] = new_sum

        for i in range(2**(self.max_depth - 1) - 2, -1, -1):
            self.tree[i] = self.tree[i * 2 + 1] + self.tree[i * 2 + 2]


class MemoryFragment:
    def __init__(self, state, action, reward, next_state, model_logit, target_logits):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.model_logit = model_logit
        self.target_logits = target_logits

    def updateLogits(self, model):
        self.model_logit = model.predict([np.expand_dims(self.state, 0), np.expand_dims(self.action, 0)])[0][0]
        return self.loss()

    def updateTargetLogits(self, target):
        if self.next_state is None:
            self.target_logits = [0., 0., 0., 0., 0.]
        else:
            self.target_logits = []
            act_arr = np.zeros(1)
            for i in range(5):
                act_arr = i * 1.0
                self.target_logits.append(target_model.predict([np.expand_dims(self.next_state, 0), np.expand_dims(self.action, 0)])[0][0])

        return self.loss()

    def loss(self):
        max_i = 0
        for i in range(1, 5):
            if self.target_logits[i] > self.target_logits[max_i]:
                max_i = i

        loss_ret = self.model_logit - (self.reward + (DISCOUNT_RATE * self.target_logits[max_i]))
        loss_ret = (loss_ret * loss_ret) / 2

        return loss_ret



class Memory:
    def __init__(self):
        self.memory = SumTree()

    def addToMemory(self, state, action, reward, next_state, model_logit, target_logits):
        fragment = MemoryFragment(state, action, reward, next_state, model_logit, target_logits)
        self.memory.insert(fragment)

    #def discountRewards(self, action_count, discount_rate = .99):
    #    discounted_rewards = np.zeros_like(self.rewards)
    #    R = 0
    #    for t in reversed(range(len(self.rewards) - action_count, len(self.rewards))):
    #        R = R * discount_rate + self.rewards[t]
    #        self.rewards[t] = R

    def sampleMemory(self, batch_size=EXPERIENCE_SAMPLE_SIZE):
        return self.memory.sample(batch_size)

    def updateLogits(self, model):
        self.memory.updateLogits(model)

    def updateTargetLogits(self, target):
        self.memory.updateTargetLogits(target)


def createModel(optimizer, loss):
    state_input = tf.keras.Input(shape=(200), name="observation_input")
    action_input = tf.keras.Input(shape=(1), name="action_input")

    l1 = tf.keras.layers.concatenate([state_input, action_input])
    l2 = tf.keras.layers.Dense(32, activation='relu')(l1)
    l3 = tf.keras.layers.Dense(32, activation='relu')(l2)
    output = tf.keras.layers.Dense(1, activation=None)(l3)

    model = tf.keras.Model(
        inputs=[state_input, action_input],
        outputs=output,
    )

    model.compile(tf.keras.optimizers.Adam(1e-4), loss)

    return model


def nextAction(model, observation):

    logit_list = []
    obs_copy = np.expand_dims(observation, 0)
    act_arr = np.array([[0.]])

    for action in range(5):
        act_arr[0][0] = ((action * 1.0) - 2) / 2
        logit_list.append(model.predict([obs_copy, act_arr])[0][0])

    return logit_list.index(max(logit_list)), max(logit_list)



def train_step(model, target, fragments):
    next_rewards = np.zeros(EXPERIENCE_SAMPLE_SIZE)

    state_list = []
    action_list = []
    Q_list = []

    for i in range(EXPERIENCE_SAMPLE_SIZE):
        state_list.append(fragments[i].state)
        action_list.append((fragments[i].action - 2.0) / 2)

        Q_i = 0

        for j in range(1, 5):
            if fragments[i].target_logits[j] > fragments[i].target_logits[Q_i]:
                Q_i = j

        Q_list.append(fragments[i].reward + DISCOUNT_RATE * fragments[i].target_logits[Q_i])

    model.fit([np.array(state_list), np.array(action_list)], np.array(Q_list), batch_size = EXPERIENCE_SAMPLE_SIZE, epochs = 5)

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

    #for i in (os.environ['PATH'].split(';')):
    #    print(i)

    #1 / 0

    board = TetrisSimulation()

    model = createModel('adam', 'mse')
    target_model = createModel('adam', 'mse')
    if os.path.isdir('./model'):
        print("Loading model...")
        model = tf.keras.models.load_model('./model')
        target_model = tf.keras.models.load_model('./model')
        print("Model loaded")

    memory = Memory()

    epsilon = 1.
    epsilon_decay_rate = .999995
    epsilon_min = .02
    sync_frequency = 10
    sync_ratio = 1.
    buffer_rate = 1.

    sync_counter = 0

    max_score = 0

    step = 0

    if os.path.isfile('./vars.txt'):
        f = open('vars.txt', 'r')
        data = f.read().split('\n')
        epsilon = float(data[0])
        step = int(data[1])
        max_score = float(data[2])

        f.close()

    supervised_step = False
    flag = True

    if supervised_step == True:
        while True:
            flag == True
            print("Supervised Learning Step:")
            board.printBoard()
            print(board.rotation_state)

            observation = np.copy(board.board).flatten()
            action = -1

            c = msvcrt.getch()
            os.system('cls')
            old_score = board.score
            if flag == False:
                break
            if c == b'a':
                action = 2
                flag = board.step("left")
            elif c == b's':
                action = 4
                flag = board.step("down")
            elif c == b'd':
                action = 3
                flag = board.step("right")
            elif c == b'w':
                action = 0
                flag = board.step("rleft")
            elif c == b'e':
                action = 0
                flag = board.step("rright")

            reward = (board.score - old_score) * 1.0

            #new_observation = np.copy(board.board).flatten()

            #target_logits = []

            #for i in range(5):
            #    target_logits.append(target_model.predict([np.expand_dims(new_observation, 0), np.array([[(i * 1. - 2.) / 2]])])[0][0])

            #if action != -1:
            #    memory.addToMemory(observation, action, reward, new_observation, logit, target_logits)

                

    if supervised_step == True:
        observations, actions, rewards = memory.sampleMemory(model, target)
        train_step(model, target_model, optimizer, observations, actions, rewards)

    #memory_buffer_full = False

    for i_episode in range(step, 5000):
        observation = np.copy(board.board).flatten()
        print("Step: ", i_episode)

        action_counter = 0

        while True:
            add_to_buffer = False
            if random.random() < buffer_rate:
                add_to_buffer = True

            if random.random() < epsilon:
                action = random.randint(0, 4)
                if add_to_buffer:
                    logit = model.predict([np.expand_dims(observation, 0), np.array([[(action * 1. - 2.) / 2]])])[0][0]
                #print("LOGIT: ", logit)
            else:
                action, logit = nextAction(model, observation)
            
            action_counter += 1

            epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)

            old_score = board.score

            continue_flag = board.step(action_map[action])
            reward = (board.score - old_score) * 1.0

            new_observation = np.copy(board.board).flatten()

            if not continue_flag:
                reward -= 100
                new_observation = None

            if add_to_buffer:

                target_logits = []
                if new_observation is None:
                    target_logits = [0., 0., 0., 0., 0.]
                else:
                    for i in range(5):
                        target_logits.append(target_model.predict([np.expand_dims(new_observation, 0), np.array([[(i * 1. - 2.) / 2]])])[0][0])

                memory.addToMemory(observation, action, reward, new_observation, logit, target_logits)

            #if (not memory_buffer_full) and len(memory.states) >= EXPERIENCE_BUFFER_SIZE:
                #print("MEMORY BUFFER AT CAPACITY, DELETING FUTURE MEMORIES")
                #memory_buffer_full = True
                #1 / 0

            if not continue_flag:
                #memory.discountRewards(action_counter)
                fragments = memory.sampleMemory()
                train_step(model, target_model, fragments)
                memory.updateLogits(model)

                #print("Step ", i_episode, " ended")
                print("Score: ", board.score)

                if board.score > max_score:
                    max_score = board.score

                print("Highscore: ", max_score)

                board.printBoard()

                board.clear()

                sync_counter += 1

                if sync_counter == sync_frequency:
                    sync_counter = 0
                    syncModels(model, target_model, sync_ratio)
                    memory.updateTargetLogits(target_model)
                    print("Epsilon: ", epsilon)
                    model.save('./model')
                    f = open('./vars.txt', 'w')
                    f.write(str(epsilon) + '\n' + str(i_episode) + '\n' + str(max_score))
                    f.close()

                break

            observation = new_observation
