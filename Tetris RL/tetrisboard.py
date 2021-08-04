import numpy as np
#import msvcrt
import random
import os

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

class tetrisboard(object):

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

    def playGame(self):
        action = -1

        flag = True

        while True:
            board.printBoard()

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

    def getObservation(self):
        return np.copy(self.board).flatten()

if __name__ == "__main__":
    print("Unit Tests for Board:")
    board = tetrisboard()

    board.clear()

    board.moveLeft()
    board.moveRight()

    board.rotateLeft()
    board.rotateRight()

    board.clear()

    board.playGame()