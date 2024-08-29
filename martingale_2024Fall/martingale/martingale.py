""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "jzheng429"  # replace tb34 with your Georgia Tech username.

def study_group():
    """
    Returns
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

    Return type
        str
    """
    return "jzheng429"

def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return 903510650  # replace with your GT ID number
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result
  		  	   		 	   		  		  		    	 		 		   		 		  
def run_episodes(n, winnings, win_prob):
    # 10 episodes, where each episode is 1000
    for i in range(n):
        episode_winnings = 0
        spins = 0
        while spins < 1000 and episode_winnings <= 80:
            won = False
            bet_amount = 1
            while not won:
                # wager bet_amount on black
                spins += 1
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings = episode_winnings + bet_amount
                else:
                    episode_winnings = episode_winnings - bet_amount
                    bet_amount = bet_amount * 2
                winnings[i][spins+1] = episode_winnings
        if spins < 1000:
            cur_winnings = winnings[i][spins]
            winnings[i][spins+1:] = cur_winnings
    return winnings

def experiment_two(n, winnings, win_prob):
    # 10 episodes, where each episode is 1000
    for i in range(n):
        episode_winnings = 0
        spins = 0
        while episode_winnings > -256 and episode_winnings <= 80:
            won = False
            bet_amount = 1
            while not won:
                # wager bet_amount on black
                spins += 1
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings = episode_winnings + bet_amount
                else:
                    if episode_winnings - bet_amount < -256:
                        episode_winnings = episode_winnings - abs(-256 - episode_winnings)
                        break
                    else:
                        episode_winnings = episode_winnings - bet_amount
                        bet_amount = bet_amount * 2
                winnings[i][spins+1] = episode_winnings
        if spins < 1000:
            cur_winnings = winnings[i][spins]
            winnings[i][spins+1:] = cur_winnings
    return winnings

def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    win_prob = 0.474  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	   		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments

    # FIGURE ONE
    winnings = np.zeros((10, 1001))
    f1_data = run_episodes(10, winnings, win_prob)
    for i in range(f1_data.shape[0]):
        plt.plot(f1_data[i], label=f'Episode {i + 1}')

    plt.title('Figure One Plot')
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.savefig('./images/figure_1.png')
    plt.clf()

    # FIGURE TWO AND THREE DATA
    winnings = np.zeros((1001, 1001))
    data = run_episodes(1000, winnings, win_prob)
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    std = np.std(data)

    # FIGURE TWO
    plt.plot(mean, label= 'mean')
    plt.plot(mean + std, label='mean + std')
    plt.plot(mean - std, label='mean - std')
    plt.title('Figure Two Plot')
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.savefig('./images/figure_2.png')
    # plt.show()
    plt.clf()

    # FIGURE THREE
    plt.plot(median, label='median')
    plt.plot(median + std, label='median + std')
    plt.plot(median - std, label='median - std')
    plt.title('Figure Three Plot')
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.savefig('./images/figure_3.png')
    # plt.show()
    plt.clf()

    # FIGURE FOUR AND FIVE DATA
    winnings = np.zeros((1001, 1001))
    data2 = experiment_two(1000, winnings, win_prob)
    mean2 = np.mean(data2, axis=0)
    median2 = np.median(data2, axis=0)
    std = np.std(data2)
    # plot_fig_one(data2)

    # FIGURE FOUR
    plt.plot(mean2, label= 'mean')
    plt.plot(mean2 + std, label='mean + std')
    plt.plot(mean2 - std, label='mean - std')
    plt.title('Figure Four Plot')
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.savefig('./images/figure_4.png')
    # plt.show()
    plt.clf()

    # FIGURE FIVE
    plt.plot(median2, label='median')
    plt.plot(median2 + std, label='median + std')
    plt.plot(median2 - std, label='median - std')
    plt.title('Figure Five Plot')
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.savefig('./images/figure_5.png')
    # plt.show()
    plt.clf()

if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
