#Class to store the game board

#Created by Robert J Dunski, unless otherwise noted in the header
###################################################################################################################################

import numpy as np
from const import Const

class GameBoard:
    def __init__(self):
        self.board = np.empty([9,9]) #intialize board with 9x9 2-d array
        self.setupBoard()
    def setupBoard(self):
        for x in range (0,9):
            for i in range(0,9):
                self.board[x][i]=Const.EMPTY #set all spaces to EMPTY

        self.board[4][4]=Const.KING          #King starts in the center

        self.board[0][0]=Const.CORNER        #Add the 4 corners
        self.board[0][8]=Const.CORNER
        self.board[8][0]=Const.CORNER
        self.board[8][8]=Const.CORNER

        for x in range(0,4):                                #Place each sides' pieces accordingly _____________________________________
            if x < 2:                                       #                                    | X |   |   | A | A | A |   |   | X |
                self.board[2+x][4]=Const.DEFENDER           #                                    |   |   |   |   | A |   |   |   |   |
                self.board[4][2+x]=Const.DEFENDER           #                                    |   |   |   |   | D |   |   |   |   |
                self.board[4][x]=Const.ATTACKER             #                                    | A |   |   |   | D |   |   |   | A |
                self.board[x][4]=Const.ATTACKER             #                                    | A | A | D | D | K | D | D | A | A |
                self.board[4][8-x]=Const.ATTACKER           #                                    | A |   |   |   | D |   |   |   | A |
                self.board[8-x][4]=Const.ATTACKER           #                                    |   |   |   |   | D |   |   |   |   |
            else:                                           #                                    |   |   |   |   | A |   |   |   |   |
                self.board[5+(-2*(x-2))][0]=Const.ATTACKER  #                                    | X |   |   | A | A | A |   |   | X |
                self.board[0][5+(-2*(x-2))]=Const.ATTACKER  #                                    _____________________________________
                self.board[5+(-2*(x-2))][8]=Const.ATTACKER
                self.board[8][5+(-2*(x-2))]=Const.ATTACKER
                self.board[8-x][4]=Const.DEFENDER
                self.board[4][8-x]=Const.DEFENDER
        #print(self.board)
