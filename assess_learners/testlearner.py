""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
import sys  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  	   		 	 	 			  		 			     			  	 
import numpy as np
import time  		  	   		 	 	 			  		 			     			  	   		  	   		 	 	 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import matplotlib.pyplot as plt

def gtid():
    return 904080678
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 			  		 			     			  	 
        sys.exit(1)

    data = np.genfromtxt(sys.argv[1], delimiter=",")
    if sys.argv[1]== "Data/Istanbul.csv":
        data = data[1:, 1:]

    # compute how much of the data is training and testing  		  	   		 	 	 			  		 			     			  	 
    sum_num=data.shape[0]
    train_size=int(sum_num*0.6)
    np.random.seed(gtid())
    train_indices=np.random.choice(sum_num,train_size,replace=False)

    data_train=data[train_indices]
    mask = np.ones(sum_num, dtype=bool)
    mask[train_indices] = False
    data_test=data[mask]
    train_x=data_train[:,:-1]
    train_y=data_train[:,-1]
    test_x=data_test[:,:-1]
    test_y=data_test[:,-1]


    """
    #Experiment1: DTLearner overfiting
    """

    #compare RMSE in leaf size from 1 to 50
    def rmse(l,X1,y1,X2,y2):
        l.add_evidence(X1, y1)
        pred1=l.query(X1)
        in_r=np.sqrt(np.mean((y1 - pred1) ** 2))
        pred2=l.query(X2)
        out_r=np.sqrt(np.mean((y2 - pred2) ** 2)) 
        return np.array([in_r,out_r])
    
    learners=np.array([dt.DTLearner(leaf_size=i+1) for i in range(50)])
    
    in_and_out_rmse=np.array([rmse(learner,train_x,train_y,test_x,test_y) for learner in learners])

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1,51), in_and_out_rmse[:, 0], label="In-Sample RMSE")
    plt.plot(np.arange(1,51), in_and_out_rmse[:, 1], label="Out-of-Sample RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 1: Effect of Leaf Size on In-Sample and Out-of-Sample RMSE")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figure1.png")


    """
    #Experiment 2
    """
    
    def dt_bag_rmse(leaf,X1,y1,X2,y2):
        dt_l=dt.DTLearner(leaf_size=leaf)
        dt_l.add_evidence(X1,y1)
        dt_pred=dt_l.query(X2)
        dt_rmse=np.sqrt(np.mean((y2 - dt_pred) ** 2))
        bl_l=bl.BagLearner(learner=dt.DTLearner,kwargs={"leaf_size":leaf},bags=20)
        bl_l.add_evidence(X1,y1)
        bl_pred=bl_l.query(X2)
        bl_rmse= np.sqrt(np.mean((y2 - bl_pred) ** 2))
        return np.array([dt_rmse,bl_rmse])
    
    rmse_dt_bag=np.array([dt_bag_rmse(i+1,train_x,train_y,test_x,test_y) for i in range(50)])

    bl_learners=np.array([bl.BagLearner(learner=dt.DTLearner,kwargs={"leaf_size":i},bags=20) for i in range(50)])
    in_and_out_bl_rmse = np.array([rmse(learner, train_x, train_y, test_x, test_y) for learner in bl_learners])
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1,51), rmse_dt_bag[:, 0], label="Single Decision Tree RMSE")
    plt.plot(np.arange(1,51), rmse_dt_bag[:, 1], label="Bagged Decision Trees (20 Bags) RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 2a: Bagging on Out-of-Sample Performance in Decision Trees")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figure2.png")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1,51), in_and_out_bl_rmse[:, 0], label="Bagged Trees (20 Bags) - In-Sample")
    plt.plot(np.arange(1,51), in_and_out_bl_rmse[:, 1], label="Bagged Trees (20 Bags) - Out-of-Sample")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 2b: In-Sample vs Out-of-Sample RMSE for Bagged Decision Trees")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figure3.png")

    """
    #Experiment 3
    """

    MAE=np.zeros((50, 2))
    train_time=np.zeros((50, 2))
    for i in range(50):
        start_time = time.time()
        dt_learner=dt.DTLearner(leaf_size=i+1)
        dt_learner.add_evidence(train_x,train_y)
        dt_y_hat=dt_learner.query(test_x)
        dt_train_time = time.time() - start_time
        
        start_time = time.time()
        rt_learner=rt.RTLearner(leaf_size=i+1)
        rt_learner.add_evidence(train_x,train_y)
        rt_y_hat=rt_learner.query(test_x)
        rt_train_time = time.time() - start_time

        #calculate MAE
        dt_mae = np.mean(np.abs(test_y - dt_y_hat))
        rt_mae = np.mean(np.abs(test_y - rt_y_hat))
        MAE[i] = [dt_mae, rt_mae]
        dt_mae_mean=np.mean(MAE.T[0][4:21])
        rt_mae_mean=np.mean(MAE.T[1][4:21])
        dt_mae_std=np.std(MAE.T[0][4:21])
        rt_mae_std=np.std(MAE.T[1][4:21])

        #Keep track of training time
        train_time[i] = [dt_train_time, rt_train_time]

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1,51), MAE[:, 0], label="Decision Tree Out-of-Sample MAE")
    plt.plot(np.arange(1,51), MAE[:, 1], label="Random Tree Out-of-Sample MAE")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.title("Experiment 3a: Out-of-Sample MAE for DT vs. RT Across Leaf Sizes")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figure4.png")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1,51), train_time[:, 0], label="Decision Tree")
    plt.plot(np.arange(1,51), train_time[:, 1], label="Random Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("Training Time")
    plt.title("DT vs. RT: Training Time Across Leaf Sizes")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figure5.png")
