#The majority of this code was NOT created by me, but instead slightly modified from this github repo
#https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/run_this.py

#Modified by Robert J Dunski
###################################################################################################################################

#This is the .py file used to run the rest of the Tablut program

from tablut import Tablut
from table import QLearningTable
from const import Const

def playGame():
    for episode in range(Const.RUN_TIME):
        # initial observation
        observation = env.reset()
        RL.updateTable(actions=list(range(env.n_actions)))

        while True:
            # fresh env
            #env.render()
            # RL choose action based on observation
            try:
                action = RL.choose_action(str(observation))
            except ValueError as err:
                print("Game has encountered an error: {0}\nExiting game...".format(err))
                break
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            RL.updateTable(actions=list(range(env.n_actions)))

            # swap observation
            observation = observation_

            if done:
                break

            # break while loop when end of this episode

    # end of game
    print('game over')

if __name__ == "__main__":
    env = Tablut()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    playGame()
    #env.after(100, update)
    #env.mainloop()
