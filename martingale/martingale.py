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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Zhe Dang (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: zdang31 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 904080678 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "zdang31"  # replace tb34 with your Georgia Tech username.


def study_group():
    return "zdang31"

  		  	   		 	 	 			  		 			     			  	 
def gtid():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return 904080678  # replace with your GT ID number
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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

def episode(win_prob,num_episodes,record):
    for i in range(num_episodes):
        episode_winnings=0
        spin = 0
        record[i,spin]=0
        while episode_winnings < 80:
            bet_amount = 1
            won = False
            while not won:
                won = get_spin_result(win_prob)
                spin+=1
                if won:
                    episode_winnings += bet_amount
                    record[i,spin]=episode_winnings
                else:
                    episode_winnings -= bet_amount
                    bet_amount *= 2
                    record[i,spin]=episode_winnings
        record[i,spin:]=episode_winnings
    return record

def real_episode(win_prob,num_episodes,record):
    for i in range(num_episodes):
        episode_winnings = 0
        spin = 0
        record[i,spin]=0
        while episode_winnings < 80:
            if episode_winnings <=-256:
                break
            bet_amount = 1
            won = False
            while not won:
                won = get_spin_result(win_prob)
                spin+=1
                if won:
                    episode_winnings += bet_amount
                    record[i,spin]=episode_winnings
                else:
                    episode_winnings -= bet_amount
                    record[i, spin] = episode_winnings
                    if episode_winnings-bet_amount*2<-256:
                        bet_amount = 256+episode_winnings
                    else:
                        bet_amount *= 2
        record[i,spin:]=episode_winnings
    return record


def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Method to test your code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    win_prob = 18/38   # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			     			  	 
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    """
    Experiment 1.1
    """
    num_episodes = 10
    num_spins = 1001
    record_1 = np.zeros((num_episodes, num_spins))
    record_1 = episode(win_prob, num_episodes, record_1)
    #Figure 1
    import matplotlib.pyplot as plt
    plt.plot(record_1.T,label = ['episode 1','episode 2','episode 3','episode 4','episode 5','episode 6','episode 7','episode 8','episode 9','episode 10'])
    plt.title("Figure 1- Total winnings in 10 episodes simulation")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Number of Spins")
    plt.ylabel("Total Winnings")
    plt.legend()
    plt.savefig("images/Figure 1.png")
    """
    Experiment 1.2
    """
    num_episodes_pro=1000
    record_2 = np.zeros((num_episodes_pro, num_spins))
    record_2 = episode(win_prob, num_episodes_pro, record_2)
    fig2_mean = np.mean(record_2,axis=0)
    fig2_std = np.std(record_2,axis=0)
    #Figure 2
    plt.figure()
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Number of Spins")
    plt.ylabel("Means with upper and lower std lines")
    plt.plot(fig2_mean, label = 'mean')
    plt.plot(fig2_mean+fig2_std, label = 'upper std line')
    plt.plot(fig2_mean-fig2_std,label = 'lower std line')
    plt.title("Figure 2- Means and std bounds in experiment 1")
    plt.legend()
    plt.savefig("images/Figure 2.png")

    """"
    Experiment 1.3
    """
    fig3_median = np.median(record_2,axis=0)
    fig3_std = fig2_std
    #Figure 3
    plt.figure()
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Number of Spins")
    plt.ylabel("Medians with upper and lower std lines")
    plt.plot(fig3_median, label = 'median')
    plt.plot(fig3_median+fig3_std, label = 'upper std line')
    plt.plot(fig3_median-fig3_std, label = 'lower std line')
    plt.title("Figure 3- Medians and std bounds in experiment 1")
    plt.legend()
    plt.savefig("images/Figure 3.png")

    """
    Experiment 2.4
    """
    record_4 = np.zeros((num_episodes_pro, num_spins))
    record_4 = real_episode(win_prob, num_episodes_pro,record_4)
    fig4_mean = np.mean(record_4,axis=0)
    fig4_std = np.std(record_4,axis=0)
    #Figure 4
    plt.figure()
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Number of Spins")
    plt.ylabel("Means with upper and lower std lines")
    plt.plot(fig4_mean, label = 'mean')
    plt.plot(fig4_mean+fig4_std, label = 'upper std line')
    plt.plot(fig4_mean-fig4_std, label = 'lower std line')
    plt.title("Figure 4- Means and std bounds in experiment 2")
    plt.legend(loc="lower left")
    plt.savefig("images/Figure 4.png")

    """
    Experiment 2.5
    """
    fig5_median = np.median(record_4,axis=0)
    fig5_std = fig4_std
    #Figure 5
    plt.figure()
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Number of Spins")
    plt.ylabel("Medians with upper and lower std lines")
    plt.plot(fig5_median, label = 'median')
    plt.plot(fig5_median+fig5_std,label = 'upper std line')
    plt.plot(fig5_median-fig5_std, label = 'lower std line')
    plt.title("Figure 5- Medians and std bounds in experiment 2")
    plt.legend()
    plt.savefig("images/Figure 5.png")
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()
    #np.savetxt("record_1.csv", df, delimiter=",")



