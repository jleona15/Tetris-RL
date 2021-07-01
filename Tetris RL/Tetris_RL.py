
import tensorflow as tf
import time
import numpy as np
from PIL import ImageGrab, Image
import pydirectinput
import os
import random
import msvcrt

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
        self.board = np.zeros((10, 20))
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


    def addGapPoints(self):
        max_index = -1

        for j in range(20):
            for i in range(10):
                if self.board[i][j] == BOARD_OCCUPIED:
                    max_index = j
                    break
            if max_index != -1:
                break

        for i in self.active_indices:
            if max_index != -1 and i[1] >= max_index:
                self.score += 1
            else:
                self.score -= 9

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
                self.addGapPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        if self.ticks_since_down >= 8:
            if not self.moveDown():
                self.addGapPoints()
                for i in self.active_indices:
                    if i[1] >= 0:
                        self.board[i[0]][i[1]] = BOARD_OCCUPIED
                ret = self.generateNewPiece()
                self.clearLines()

        return ret

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
        tf.keras.layers.Conv2D(filters=64, kernel_size=(10, 1), activation='relu', input_shape=(10, 20, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation=None)
    ])

    return model


def nextAction(model, observation):
    logits = model.predict(observation)

    return logits.argmax()


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


def discount_rewards(rewards, discount_rate = 1.):
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
    print(observations.shape)

    with tf.GradientTape() as tape:
        logits = model.__call__(observations)

        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def performAction(key):
    pydirectinput.keyDown(key)
    pydirectinput.keyDown('n')
    pydirectinput.keyUp(key)
    pydirectinput.keyUp('n')


if __name__ == "__main__":

    board = TetrisSimulation()

    while True:
        board.printBoard()
        print(board.rotation_state)
        c = msvcrt.getch()
        os.system('cls')
        if c == b'a':
            board.step("left")
        elif c == b's':
            board.step("down")
        elif c == b'd':
            board.step("right")
        elif c == b'w':
            board.step("rleft")
        elif c == b'e':
            board.step("rright")

    model = createModel()
    memory = Memory()

    #1 / 0

    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    if os.path.isdir("model"):
        model.load("model")

    #if hasattr(tqdm, '_instances'): tqdm._instances.clear()
    for i_episode in range(5000):
        observation = np.expand_dims(np.copy(board.board), axis=-1)

        memory.clear()

        print("Step: ", i_episode)

        #count = 0

        while True:
            action = nextAction(model, np.expand_dims(np.copy(observation), axis=(0, -1)))
            #count += 1
            #print(action)

            #if (count % 500) == 0:
            #    board.printBoard()

            #if count > 2000:
            #    board.printBoard()
            #    print(action)
            #    print(board.ticks_since_down)

            old_score = board.score

            continue_flag = board.step(action_map[action])
            reward = (board.score - old_score) * 1.0

            new_observation = np.expand_dims(np.copy(board.board), axis=-1)

            memory.addToMemory(observation, action, reward)

            if not continue_flag:
                train_step(model, optimizer,
                           observations=np.array(memory.observations),
                           actions=np.array(memory.actions),
                           discounted_rewards=discount_rewards(memory.rewards))

                #print("Step ", i_episode, " ended")
                print(board.score)
                memory.clear()
                board.clear()
                break


            observation = new_observation

    model.save('model')