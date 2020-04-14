#NOTE: Some TKinter portions, the update function, and main function (at the very bottom) in this file were originally obtained from the following github repo:
#https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/maze_env.py


#Created by Robert J Dunski, unless otherwise noted in the header
###################################################################################################################################
#Main python file for the game environment, Tablut. Game rules are listed below, source: http://tafl.cyningstan.com/page/170/tablut


# 1. Tablut is played on a board of 9×9 squares.
#
# 2. There are 25 pieces: a king and his eight defenders, and sixteen attackers. These are placed in the shape of a cross with serifs, as in the diagram (on website).
#
# 3. The attacking side takes the first move.
#
# 4. Pieces move any distance orthogonally, not landing on nor jumping over other pieces on the board.
#
# 5. No piece may land on the central square, called the "castle", not even the king once he has left it.
#
# 6. A piece other than the king is captured when it is surrounded orthogonally on two opposite squares by enemies. The king can pair up with a defender for the purpose of capturing attackers.
#
# 7. A piece may also be captured between an enemy and the empty castle.
#
# 8. When in the castle, the king is captured by surrounding him on four orthogonal sides with attackers.
#
# 9. When stood beside the castle, the king may be captured by surrounding him on the remaining three sides with attackers.
#
# 10. Elsewhere on the board, the king is captured as other pieces.
#
# 11. If the king when in the castle is surrounded on three sides by attackers, and on the fourth by a defender, the defender may be captured by surrounding it between an attacker and the king.
#
# 12. The king wins the game on reaching any square at the edge of the board. The attackers win if they capture the king.
#
# 13. The game is drawn if a position is repeated, if a player cannot move, or if the players otherwise agree it.

import numpy as np
import random
import time
from const import Const
from table import QLearningTable
from board import GameBoard
#import tkinter as tk

class Move:
    def __init__(self,piece,fromRow,fromCol,toRow,toCol):
        self._piece=piece
        self._fromRow=fromRow
        self._toRow=toRow
        self._fromCol=fromCol
        self._toCol=toCol

class Tablut(object):
    def __init__(self):
        super(Tablut, self).__init__()
        self.att_reward_total=0                                                 #Following variable are used in printStats()
        self.def_reward_total=0
        self.ATT_WIN_COUNT=0
        self.DEF_WIN_COUNT=0
        self.DEF_CAPTURED=0
        self.ATT_CAPTURED=0
        self._turns=0
        self._state=Const.ATT_TURN                                              #See rule #3 of game rules
        self.done=False                                                         #Used to stop game

        self.captured=[]                                                        #Keep track of captured pieces

        self._board=GameBoard()                                                 #Instantiate board
        self.action_space=self.newMovements(Const.ATT_SIDE_MARK)                                                  #Holds all possible moves for either side

        self.n_actions = len(self.action_space)                                 #Used for QLearningTable algorithm
        #self.title('tablut')
        #self.geometry('{0}x{1}'.format(UNIT*9, UNIT*9))
        # create origin
        #self.center = np.array([180, 180])
        #self.buildGame()
    def resetGame(self):                                                        #Reset Game (could probably do this more graciously)
        self.ATT_CAPTURED=0
        self.DEF_CAPTURED=0
        self.att_reward_total=0
        self.def_reward_total=0
        self.captured=[]
        self.done=False
        self._board=GameBoard()
        self.action_space = []
        self._turns=0
        self._state=Const.ATT_TURN
        self.action_space=self.newMovements(Const.ATT_SIDE_MARK)
        self.n_actions = len(self.action_space)

    def newMovements(self,side):                                                #Retrieves all possible moves for either side
        action_space = []
        for row in range(0,9):
            for col in range(0,9):

                if side==Const.ATT_SIDE_MARK:
                    if self._board.board[row][col] == Const.ATTACKER:
                        for dist in range (1,9):                                #Pieces shouldn't be able to move more than 8
                            action_space.append(self.movements(row,col,dist))   #spaces or less than 1 space in any direction

                else:                                                           #regardless of location on the board
                    if self._board.board[row][col] == Const.DEFENDER \
                    or self._board.board[row][col] == Const.KING:               #Defenders have 2 types of moveable pieces
                        for dist in range (1,9):
                            action_space.append(self.movements(row,col,dist))

        action_space=[i for i in action_space if i]                             #Remove None values and empty values

        return action_space

    def movements(self, row,col, dist):                                         #Sub-function for newMovements() to step through all
        piece=self._board.board[row][col]                                       #possible moves
        moves = []

        for x in Const.DIRS:
            toRow=row+(dist*x[0])                                               #check dist and direction 4 times for each moveset
            toCol=col+(dist*x[1])

            if (toRow > 8 or toCol > 8) or (toRow < 0 or toCol < 0):            #Piece can't move off the board
                continue
            try:
                move=Move(piece,row,col,toRow,toCol)
                self.moveOk(move)                                               #Check is move is in fact possible
                moves.append(move)                                              #If so, add it to the array
            except ValueError:
                continue                                                        #If not, move on

        return moves

    def moveOk(self,move):                                                      #Sub-function for movements()
        if move._piece != Const.ATTACKER\
        and move._piece != Const.DEFENDER\
        and move._piece != Const.KING:                                          #Can't move EMPTY,CORNER,or CENTER "pieces"
            raise ValueError("cannot move invalid piece")                       #Check for illegal moves, exception handling details

        if move._toRow < 0 or move._toCol < 0:                                  #the type of invalidity
            raise ValueError("negative moves not allowed")

        if move._fromCol != move._toCol and move._fromRow != move._toRow:
            raise ValueError("cannot move diagonal")

        if self._board.board[move._toRow][move._toCol] != Const.EMPTY:
            raise ValueError("destination (to) is occupied")

        if move._toCol == 4 and move._toRow == 4:
            raise ValueError("cannot move to center")

        if self._board.board[move._fromRow][move._fromCol] != move._piece:
            raise ValueError("source (from) is not moveable")

        if move._fromRow != move._toRow:                                        #piece is moving up or down

            if move._toRow-move._fromRow < 0:                                   #check for "backwards" movements
                for x in range(move._toRow,move._fromRow-1):
                    spaceInBetween=self._board.board[x][move._fromCol]
                    if spaceInBetween != Const.EMPTY:
                        raise ValueError("cannot move over pieces or center")

            else:
                for x in range(move._fromRow+1,move._toRow):
                    spaceInBetween=self._board.board[x][move._fromCol]
                    if spaceInBetween != Const.EMPTY:
                        raise ValueError("cannot move over pieces or center")

        if move._fromCol != move._toCol:                                        #piece is moving left or right

            if move._toCol-move._fromCol < 0:                                   #check for "backwards" movements
                for x in range(move._toCol,move._fromCol-1):
                    spaceInBetween=self._board.board[move._fromRow][x]
                    if spaceInBetween != Const.EMPTY:
                        raise ValueError("cannot move over pieces or center")

            else:
                for x in range(move._fromCol+1,move._toCol):
                    spaceInBetween=self._board.board[move._fromRow][x]
                    if spaceInBetween != Const.EMPTY:
                        raise ValueError("cannot move over pieces or center")

    def play(self,move):                                                        #Function to move pieces around the board
        if self.done:
            raise RuntimeError("move after game is over")
        self._turns = self._turns + 1

        if move._piece == Const.KING and self._board.board[4][4] == Const.KING: #See rule #5 of game rules
            self._board.board[move._fromRow][move._fromCol]=Const.CENTER
        else:
            self._board.board[move._fromRow][move._fromCol]=Const.EMPTY         #Change "from" space to EMPTY

        self._board.board[move._toRow][move._toCol]=move._piece                 #Move piece to space

        if self._state == Const.DEF_TURN:                                       #Defender turn logic and win conditions
            for x in Const.DIRS:
                try:
                                                                                                         #If DEFENDER moves next to an ATTACKER
                    if self._board.board[move._toRow+(1*x[0])][move._toCol+(1*x[1])] == Const.ATTACKER\
                    and move._toRow+(1*x[0]) >= 0 and move._toCol+(1*x[1]) >= 0:                                #and not negative indexing

                                                                                                                #If the space across from the ATTACKER is an allied piece or the CENTER
                        if (self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.DEFENDER\
                        or self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.CENTER\
                        or self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.KING)\
                        and move._toRow+(2*x[0]) >= 0 and move._toCol+(2*x[1]) >= 0:                            #and not negative indexing

                            self.captured.append(Const.ATTACKER)                                                #ATTACKER is captured
                            self._board.board[move._toRow+(1*x[0])][move._toCol+(1*x[1])] = Const.EMPTY

                            self.def_reward=1                                                                   #Reward DEFENDER for capturing
                            self.att_reward=-1                                                                  #Punish ATTACKER for losing a piece

                            self.def_reward_total= self.def_reward_total+1
                            self.att_reward_total= self.att_reward_total-1

                except IndexError:
                    continue

            for x in range(0,9):
                try:
                                                                                #If the king is on ANY of the edges, DEFENDER Wins
                    if self._board.board[x][0] == Const.KING \
                    or self._board.board[0][x] == Const.KING \
                    or self._board.board[x][8] == Const.KING \
                    or self._board.board[8][x] == Const.KING:

                        self._state=Const.DEF_WIN
                        self.done=True

                        self.def_reward = 5                                     #Reward DEFENDER for winning
                                                                                #But don't punish ATTACKER for losing (testing learning changes)
                        self.def_reward_total= self.def_reward_total+5

                except IndexError:
                    continue

            if not self.done:
                self._state = Const.ATT_TURN                                    #Switch turns
                return

        elif self._state == Const.ATT_TURN:                                     #ATTACKER logic and win conditions
            for x in Const.DIRS:                                                #WARNING: Lots of math ahead
                try:
                                                                                                                    #If ATTACKER moves next to DEFENDER
                    if self._board.board[move._toRow+(1*x[0])][move._toCol+(1*x[1])] == Const.DEFENDER\
                    and move._toRow+(1*x[0]) >= 0 and move._toCol+(1*x[1]) >= 0:                                    #and not negative indexing

                                                                                                                    #If the space across from the DEFENDER is another ATTACKER
                        if self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.ATTACKER\
                        or self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.CENTER\
                        and move._toRow+(2*x[0]) >= 0 and move._toCol+(2*x[1]) >= 0:                                #and not negative indexing
                            self.captured.append(Const.DEFENDER)                                                    #DEFENDER is captured
                            self._board.board[move._toRow+(1*x[0])][move._toCol+(1*x[1])] = Const.EMPTY
                            self.def_reward=-1                                                                      #Punish DEFENDER for losign a piece
                            self.att_reward=1                                                                       #Reward ATTACKER for capturing

                            self.att_reward_total= self.att_reward_total+1
                            self.def_reward_total= self.def_reward_total-1
                                                                                                                    #If ATTACKER moves next to KING
                    elif self._board.board[move._toRow+(1*x[0])][move._toCol+(1*x[1])] == Const.KING\
                    and move._toRow+(1*x[0]) >= 0 and move._toCol+(1*x[1]) >= 0:                                    #and not negative indexing

                                                                                                                    #First, check if the space across is the CENTER
                        if self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.CENTER\
                        and move._toRow+(2*x[0]) >= 0 and move._toCol+(2*x[1]) >= 0:                                #and not negative indexing

                            king_row=move._toRow+(1*x[0])                                                           #Get the coordinates of the KING
                            king_col=move._toCol+(1*x[1])

                            self.att_reward=self.att_reward+1                                                       #Reward the ATTACKER prematurely to
                                                                                                                    #incentivise getting close to the KING
                                                                                                                    #If the KING has another ATTACKER next to it
                            if self._board.board[king_row+(1*x[1])][king_col+(1*x[0])] == Const.ATTACKER\
                            and king_row+(1*x[1]) >= 0 and king_col+(1*x[0]) >= 0:                                  #and not negative indexing

                                surround_piece_row=king_row+(1*x[1])                                                #Get the coordinates of the other ATTACKER
                                surround_piece_col=king_col+(1*x[0])                                                #next to the king
                                                                                                                                    #If there is another ATTACKER across from the surrounding piece
                                if self._board.board[surround_piece_row+(-2*x[1])][surround_piece_col+(-2*x[0])] == Const.ATTACKER\
                                and surround_piece_row+(2*x[1]) >= 0 and surround_piece_col+(2*x[0]) >= 0:

                                    self._state=Const.ATT_WIN                                                                       #The KING is surrounded, so ATTACKER Wins
                                    self.done=True
                                    self.att_reward=self.att_reward+5                                                               #Reward ATTACKER for winning
                                    self.att_reward_total= self.att_reward_total+5

                        elif self._board.board[4][4] == Const.KING:                                                                 #Second, if the KING is currently on the CENTER
                                                                                                                                    #Check each surrounding piece for ATTACKERs
                            if self._board.board[4][5] == Const.ATTACKER and self._board.board[4][3] == Const.ATTACKER\
                            and self._board.board[5][4] == Const.ATTACKER and self._board.board[3][4] == Const.ATTACKER:

                                self._state=Const.ATT_WIN                                                                           #KING is surrounded, ATTACKER Wins
                                self.done=True
                                self.att_reward=self.att_reward+5
                                self.att_reward_total= self.att_reward_total+5

                        elif self._board.board[4][4] == Const.CENTER:
                                                                                                                                    #Finally, check if the KING is not next to the CENTER
                            if self._board.board[4][5] != Const.KING and self._board.board[4][3] != Const.KING\
                            and self._board.board[5][4] != Const.KING and self._board.board[3][4] != Const.KING:
                                                                                                                                    #If so, king is captured like any other piece,
                                    if self._board.board[move._toRow+(2*x[0])][move._toCol+(2*x[1])] == Const.ATTACKER\
                                    and move._toRow+(2*x[0]) >= 0 and move._toCol+(2*x[1]) >= 0:                                    #See rule #10 of game rules
                                        self._state=Const.ATT_WIN
                                        self.done=True
                                        self.att_reward=self.att_reward+5
                                        self.att_reward_total= self.att_reward_total+5
                except IndexError:
                    continue

            if not self.done:
                self._state = Const.DEF_TURN                                    #Switch Turns
                return

    def printStats(self,winner):
        print("{} Win".format(winner))
        print("Attacker Total Wins: {0} Defender Total Wins: {1}".format(self.ATT_WIN_COUNT,self.DEF_WIN_COUNT))
        print("Attackers captured (this game): {0} Defenders captured (this game): {1}".format(self.ATT_CAPTURED,self.DEF_CAPTURED))
        print("Game Over in {0} turns".format(self._turns))
        print("Total Attacker Reward: {0} Total Defender Reward: {1}".format(self.att_reward_total,self.def_reward_total))

    def reset(self):                                                            #Update stats and restart the game
        if self._state==Const.ATT_WIN:
            self.ATT_WIN_COUNT=self.ATT_WIN_COUNT+1
            for x in self.captured:
                if x == Const.DEFENDER:
                    self.DEF_CAPTURED=self.DEF_CAPTURED+1
                elif x == Const.ATTACKER:
                    self.ATT_CAPTURED=self.ATT_CAPTURED+1
            self.printStats("Attackers")
        elif self._state==Const.DEF_WIN:
            self.DEF_WIN_COUNT=self.DEF_WIN_COUNT+1
            for x in self.captured:
                if x == Const.DEFENDER:
                    self.DEF_CAPTURED=self.DEF_CAPTURED+1
                elif x == Const.ATTACKER:
                    self.ATT_CAPTURED=self.ATT_CAPTURED+1
            self.printStats("Defenders")
        self.resetGame()

        return self.action_space

    def step(self, action):
        self.att_reward=0                                                       #Reset rewards for each step
        self.def_reward=0
        reward=0

        if self.done:                                                           #The terminal state is used to indicate either
            s_ = 'terminal'                                                     #the game is ending, or there is nothing to gain
                                                                                #from the next state
        else:
            actions_len = len(self.action_space)                                #Do some numpy array gymnastics to get a valid action
            try:
                self.play(self.action_space[action][0])
            except IndexError:
                try:
                    self.play(self.action_space[action])
                except:
                    self.play(self.action_space[np.random.choice(actions_len)][0])

            if self._state==Const.DEF_TURN:
                reward=self.def_reward                                          #Save the reward for use in the QLearningTable algorithm
                self.action_space = self.newMovements(Const.DEF_SIDE_MARK)      #Update moves for the DEFENDER side
                self.n_actions = len(self.action_space)
                try:
                    s_ = random.choice(self.action_space)                       #The next state will be a random piece
                except:                                                         #And the "best" move will be chosen later
                    s_='terminal'

            elif self._state==Const.ATT_TURN:
                reward=self.att_reward
                self.action_space=self.newMovements(Const.ATT_SIDE_MARK)
                self.n_actions = len(self.action_space)
                try:
                    s_ = random.choice(self.action_space)                       #Update moves for the ATTACKER side
                except:
                    s_='terminal'

            else:
                s_='terminal'                                                   #If either side wins, the next move doesn't contribute
                                                                                #to the learner
        return s_, reward, self.done

#This section is a non-implemeted graphical representation of the game using the TKinter library

##################################################################################################################

    # def render(self):
    #     time.sleep(0.1)
    #     self.update()

    # def update():
    #     for t in range(10):
    #         s = env.reset()
    #         while True:
    #             #env.render()
    #             a = 1
    #             s, r, self.done = env.step(a)
    #             if self.done:
    #                 break

    # if __name__ == '__main__':
    #     env = Tablut()
    #     env.after(100, update)
    #     env.mainloop()

    # def buildGame(self):
    #         self.canvas = tk.Canvas(self, bg='grey',
    #                            height=9 * UNIT,
    #                            width=9 * UNIT)
    #
    #         # create grids
    #         for c in range(0, 9 * UNIT, UNIT):
    #             x0, y0, x1, y1 = c, 0, c, 9 * UNIT
    #             self.canvas.create_line(x0, y0, x1, y1)
    #         for r in range(0, 9 * UNIT, UNIT):
    #             x0, y0, x1, y1 = 0, r, 9 * UNIT, r
    #             self.canvas.create_line(x0, y0, x1, y1)
    #         print(self._board.board)
    #
    #
    #         self.corner_array =[0,0,0,0]
    #         self.corner_array[0]=self.canvas.create_rectangle(-40,-40,40,40,fill='black')        #Create Corners
    #         self.corner_array[1]=self.canvas.create_rectangle(360,360,320,320,fill='black')
    #         self.corner_array[2]=self.canvas.create_rectangle(360,-40,320,40,fill='black')
    #         self.corner_array[3]=self.canvas.create_rectangle(-40,360,40,320,fill='black')
    #
    #         self.buildDefenders()
    #         self.buildAttackers()
    #         # hell2_center = origin + np.array([UNIT, UNIT * 2])
    #         # self.hell2 = self.canvas.create_rectangle(
    #         #     hell2_center[0] - 15, hell2_center[1] - 15,
    #         #     hell2_center[0] + 15, hell2_center[1] + 15,
    #         #     fill='black')
    #
    #         # create oval
    #         # oval_center = origin + UNIT * 2
    #         # self.oval = self.canvas.create_oval(
    #         #     oval_center[0] - 15, oval_center[1] - 15,
    #         #     oval_center[0] + 15, oval_center[1] + 15,
    #         #     fill='yellow')
    #
    #         # create red rect
    #
    #         # pack all
    #         self.canvas.pack()
    # def buildAttackers(self):
    #     self.attacker_array = [0]*16
    #     attacker_count=0
    #     for x in range (0,16):
    #         if attacker_count < 2:
    #             position_offset = np.array([0,UNIT*(4-x)])
    #             position = self.center - position_offset
    #         elif attacker_count < 4:
    #             position_offset = np.array([UNIT+(2*((-1*UNIT)*(x-2))),UNIT*4])
    #             position = self.center - position_offset
    #         elif attacker_count < 6:
    #             position_offset = np.array([UNIT*(4-(x-4)),0])
    #             position = self.center + position_offset
    #         elif attacker_count < 8:
    #             position_offset = np.array([UNIT*4,UNIT+(2*((-1*UNIT)*((x-4)-2)))])
    #             position = self.center + position_offset
    #         elif attacker_count < 10:
    #             position_offset = np.array([0,UNIT*(4-(x-8))])
    #             position = self.center + position_offset
    #         elif attacker_count < 12:
    #             position_offset = np.array([UNIT+(2*((-1*UNIT)*((x-8)-2))),UNIT*4])
    #             position = self.center + position_offset
    #         elif attacker_count < 14:
    #             position_offset = np.array([UNIT*(4-(x-12)),0])
    #             position = self.center - position_offset
    #         else:
    #             position_offset = np.array([UNIT*4,UNIT+(2*((-1*UNIT)*((x-12)-2)))])
    #             position = self.center - position_offset
    #         if self.attacker_array[x]:      #Check to see if there is a piece already in index
    #             print("There is already a defender at index {0} in attacker_array".format(x));
    #             exit(1);
    #         else:                           #Create piece and add it to array at index x
    #             self.attacker_array[x]= self.canvas.create_oval(
    #             position[0]-15, position[1] - 15,
    #             position[0]+15, position[1] + 15,
    #             fill='black')
    #         attacker_count = attacker_count+1
    #
    # def buildDefenders(self):
    #     self.king = self.canvas.create_oval(       #King starts in the center
    #         self.center[0] - 15, self.center[1] - 15,
    #         self.center[0] + 15, self.center[1] + 15,
    #         fill='yellow')
    #
    #     self.defender_array = [0]*9         #initialize empty array
    #     self.defender_array[8]=self.king    #add king the end of array
    #     defender_count=0
    #     for x in range(0,8):                # place defenders around the king
    #
    #         if x < 4:                       #build left, then right
    #             if defender_count < 2:
    #                 position_offset = np.array([UNIT*(x+1), 0])
    #                 position = self.center - position_offset
    #             else:
    #                 position_offset = np.array([UNIT*((x-2)+1), 0])
    #                 position = self.center + position_offset
    #
    #         else:                           #build up, then down
    #             if defender_count < 6:
    #                 position_offset = np.array([0,UNIT*((x-4)+1)])
    #                 position = self.center - position_offset
    #             else:
    #                 position_offset = np.array([0,UNIT*(((x-4)-2)+1)])
    #                 position = self.center + position_offset
    #
    #         if self.defender_array[x]:      #Check to see if there is a piece already in index
    #             print("There is already a defender at index {0} in defender_array".format(x));
    #             exit(1);
    #         else:                           #Create piece and add it to array at index x
    #             self.defender_array[x]= self.canvas.create_oval(
    #             position[0]-15, position[1] - 15,
    #             position[0]+15, position[1] + 15,
    #             fill='white')
    #
    #         defender_count= defender_count + 1  #Keep track of the number of pieces to determine placing order
    #



    ##############################################  This goes in the step() function
    # s = self.canvas.coords(piece)
    # base_action = np.array([0, 0])
    # if action == 0:   # up
    #     if s[1] > UNIT:
    #         base_action[1] -= UNIT*dist
    # elif action == 1:   # down
    #     if s[1] < (9 - 1) * UNIT:
    #         base_action[1] += UNIT*dist
    # elif action == 2:   # right
    #     if s[0] < (9 - 1) * UNIT:
    #         base_action[0] += UNIT
    # elif action == 3:   # left
    #     if s[0] > UNIT:
    #         base_action[0] -= UNIT

    #self.canvas.move(piece, base_action[0], base_action[1])  # move agent

    ##########################################  This goes in the reset() function
    #self.update()
    #time.sleep(0.5)
    # self.canvas.delete(self.king)
    # for defend in self.defender_array:
    #     self.canvas.delete(defend)
    # for att in self.attacker_array:
    #     self.canvas.delete(att)
    # self.buildDefenders()
    # self.buildAttackers()
