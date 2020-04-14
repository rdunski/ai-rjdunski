
#Created by Robert J Dunski, unless otherwise noted in the header
###################################################################################################################################

class Const:
    UNIT=40                             #Used by TKinter for graphical scaling

    RUN_TIME = 100                      #How many times to play the game

    DIRS=[(-1,0),(1,0),(0,-1),(0,1)]    #Vector array for possible moves

    EMPTY = 0                           #Empty space
    ATTACKER = 1
    KING = 2
    DEFENDER = 3
    CORNER = 4                          #The 4 corners of the board are not allowed to have pieces of any kind
    CENTER = 5                          #Once the King moves out the Center, it acts like a special corner, able to be used by either side to capture peices

    ATT_TURN=0                          #State indicators
    DEF_TURN=1
    ATT_WIN=2
    DEF_WIN=3

    ATT_SIDE_MARK=0                     #Easier handling of side pieces
    DEF_SIDE_MARK=1
