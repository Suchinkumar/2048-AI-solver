from random import randint
import time
import csv
import numpy as np
import math
from numba import jit

# Game Board **********************************************************************************************


dirs = [UP, DOWN, LEFT, RIGHT] = range(4)

@jit(nopython=True)
def merge(a):
    for i in [0,1,2,3]:
        for j in [0,1,2]:
            if a[i][j] == a[i][j + 1] and a[i][j] != 0:
                a[i][j] *= 2
                a[i][j + 1] = 0   
    return a

@jit(nopython=True)
def justify_left(a, out):
    for i in [0,1,2,3]:
        c = 0
        for j in [0,1,2,3]:
            if a[i][j] != 0:
                out[i][c] = a[i][j]
                c += 1
    return out

@jit(nopython=True)
def get_available_from_zeros(a):
    uc, dc, lc, rc = False, False, False, False

    v_saw_0 = [False, False, False, False]
    v_saw_1 = [False, False, False, False]

    for i in [0,1,2,3]:
        saw_0 = False
        saw_1 = False

        for j in [0,1,2,3]:

            if a[i][j] == 0:
                saw_0 = True
                v_saw_0[j] = True

                if saw_1:
                    rc = True
                if v_saw_1[j]:
                    dc = True

            if a[i][j] > 0:
                saw_1 = True
                v_saw_1[j] = True

                if saw_0:
                    lc = True
                if v_saw_0[j]:
                    uc = True

    return [uc, dc, lc, rc]

class GameBoard:
    def __init__(self):
        self.grid = np.zeros((4, 4))#, dtype=np.int_)

    def clone(self):
        grid_copy = GameBoard()
        grid_copy.grid = np.copy(self.grid)
        return grid_copy

    def insert_tile(self, pos, value):
        self.grid[pos[0]][pos[1]] = value

    def get_available_cells(self):
        cells = []
        for x in range(4):
            for y in range(4):
                if self.grid[x][y] == 0:
                    cells.append((x,y))
        return cells

    def get_max_tile(self):
        return np.amax(self.grid)

    def move(self, dir, get_avail_call = False):
        if get_avail_call:
            clone = self.clone()

        z1 = np.zeros((4, 4))#, dtype=np.int_)
        z2 = np.zeros((4, 4))#, dtype=np.int_)

        if dir == UP:
            self.grid = self.grid[:,::-1].T
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid.T[:,::-1]
        if dir == DOWN:
            self.grid = self.grid.T[:,::-1]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1].T
        if dir == LEFT:
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
        if dir == RIGHT:
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]

        if get_avail_call:
            return not (clone.grid == self.grid).all()
        else:
            return None

    def get_available_moves(self, dirs = dirs):
        available_moves = []
        
        a1 = get_available_from_zeros(self.grid)

        for x in dirs:
            if not a1[x]:
                board_clone = self.clone()

                if board_clone.move(x, True):
                    available_moves.append(x)

            else:
                available_moves.append(x)

        return available_moves

    def get_cell_value(self, pos):
        return self.grid[pos[0]][pos[1]]




# Main AI **********************************************************************************************

UP, DOWN, LEFT, RIGHT = range(4)

class AI():

    def get_move(self, board):
        best_move, _ = self.maximize(board)
        return best_move

    def eval_board(self, board, n_empty): 
        grid = board.grid

        utility = 0
        smoothness = 0

        big_t = np.sum(np.power(grid, 2))
        s_grid = np.sqrt(grid)
        smoothness -= np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
        smoothness -= np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
        smoothness -= np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
        smoothness -= np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
        smoothness -= np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
        smoothness -= np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
        
        empty_w = 100000
        smoothness_w = 3

        empty_u = n_empty * empty_w
        smooth_u = smoothness ** smoothness_w
        big_t_u = big_t

        utility += big_t
        utility += empty_u
        utility += smooth_u

        return (utility, empty_u, smooth_u, big_t_u)

    def maximize(self, board, depth = 0):
        moves = board.get_available_moves()
        moves_boards = []

        for m in moves:
            m_board = board.clone()
            m_board.move(m)
            moves_boards.append((m, m_board))

        max_utility = (float('-inf'),0,0,0)
        best_direction = None

        for mb in moves_boards:
            utility = self.chance(mb[1], depth + 1)

            if utility[0] >= max_utility[0]:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def chance(self, board, depth = 0):
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        # if n_empty >= 7 and depth >= 6:
        #     return self.eval_board(board, n_empty)

        if n_empty >= 6 and depth >= 3:
            return self.eval_board(board, n_empty)

        if n_empty >= 0 and depth >= 5:
            return self.eval_board(board, n_empty)

        if n_empty == 0:
            _, utility = self.maximize(board, depth + 1)
            return utility

        possible_tiles = []

        chance_2 = (.9 * (1 / n_empty))
        chance_4 = (.1 * (1 / n_empty))
        
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, chance_2))
            possible_tiles.append((empty_cell, 4, chance_4))

        utility_sum = [0, 0, 0, 0]

        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.maximize(t_board, depth + 1)

            for i in range(4):
                utility_sum[i] += utility[i] * t[2]

        return tuple(utility_sum)



#  Main GUI **********************************************************************************************


dirs = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class CLIRunner:
    def __init__(self):
        self.board = GameBoard()
        self.ai = AI()


        self.init_game()
        self.print_board()

        self.run_game()

        self.over = False

    def init_game(self):
        self.insert_random_tile()
        self.insert_random_tile()

    def run_game_for_time(self, time_limit):
        start_time = time.time()

        while True:
            self.run_game()

            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                break

            self.__init__()

    # def run_game(self):
    #     while True:
    #         move = self.ai.get_move(self.board)
    #         if move is None:
    #             print("No valid moves left. Game over.")
    #             break

    #         # print("Player's Turn:", end="")
    #         self.board.move(move)
    #         # print(dirs[move])
    #         # self.print_board()
    #         # print("Computer's Turn")
    #         self.insert_random_tile()
    #         # self.print_board()

    #         if len(self.board.get_available_moves()) == 0:
    #             max_tile = str(self.board.get_max_tile())
    #             print("GAME OVER (max tile): " + max_tile)
    #             with open('game_log.csv', 'a', newline='') as file:
    #                 writer = csv.writer(file)
    #                 writer.writerow([max_tile])
    #             self.games_completed += 1
    #             print(f"Number of games completed: {self.games_completed}")
    #             break

    def run_game(self):
        while True:
            move = self.ai.get_move(self.board)
            if move is None:
                print("No valid moves left. Game over.")
                break

            self.board.move(move)
            self.insert_random_tile()
            print(dirs[move])
            self.print_board()

            if len(self.board.get_available_moves()) == 0:
                max_tile = str(self.board.get_max_tile())
                print("GAME OVER (max tile): " + max_tile)
                
                # Read the last game count from the CSV file
                try:
                    with open('game_log.csv', 'r', newline='') as file:
                        last_line = list(csv.reader(file))[-1]
                        game_count = int(last_line[0]) + 1
                except (IndexError, FileNotFoundError):
                    game_count = 1

                # Write the new game count and max tile to the CSV file
                with open('game_log.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([game_count, max_tile])
                
                print(f"Number of games completed: {game_count}")
                break

    def print_board(self):
        for i in range(4):
            for j in range(4):
                print("%6d  " % self.board.grid[i][j], end="")
            print("")
        print("")

    def insert_random_tile(self):
        if randint(0,99) < 100 * 0.9:
            value = 2
        else:
            value = 4

        cells = self.board.get_available_cells()
        pos = cells[randint(0, len(cells) - 1)] if cells else None

        if pos is None:
            return None
        else:
            self.board.insert_tile(pos, value)
            return pos




if __name__ == '__main__':
    runner = CLIRunner()
    runner.run_game_for_time(30)  # Run the game for 30 seconds
