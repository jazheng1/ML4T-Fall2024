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
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
import time
  		  	   		 	   		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		  	   		 	   		  		  		    	 		 		   		 		  
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)
    data = []

    if sys.argv[1] == "Data/Istanbul.csv":
        with open(sys.argv[1], 'r') as file:
            lines = file.readlines()

        for line in lines[1:]:
            row = line.strip().split(',')[1:]  # Skip the first column
            data.append(row)

        # Convert the list of strings to a NumPy array and change data type if necessary
        data = np.array(data, dtype=float)  # Change dtype as needed
    else:
        inf = open(sys.argv[1])
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )

    # compute how much of the data is training and testing  		  	   		 	   		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])
    shuffled_indices = np.random.permutation(data.shape[0])

    # Select training indices
    train_indices = shuffled_indices[:train_rows]
    # Select testing indices
    test_indices = shuffled_indices[train_rows:]

    # Separate out training and testing data
    train_x = data[train_indices, :-1]  # Features for training
    train_y = data[train_indices, -1]  # Target for training
    test_x = data[test_indices, :-1]  # Features for testing
    test_y = data[test_indices, -1]  # Target for testing

    # test_rows = data.shape[0] - train_rows
    #
    # # separate out training and testing data
    # train_x = data[:train_rows, 0:-1]
    # train_y = data[:train_rows, -1]
    # test_x = data[train_rows:, 0:-1]
    # test_y = data[train_rows:, -1]
  	#
    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it  		  	   		 	   		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    dlearner = dt.DTLearner(leaf_size = 1, verbose= False)
    rlearner = rt.RTLearner(leaf_size = 1, verbose= False)
    blearner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':1}, bags=10, boost=False, verbose=False)
    dlearner.add_evidence(train_x, train_y)
    # blearner.add_evidence(train_x, train_y)
    # print(learner.author())

  	#EXPERIMENT 1
    leafs = np.arange(50, 0, -1)
    insamrmse = []
    outsamrmse = []
    for l in leafs:
        dlearner = dt.DTLearner(leaf_size=l, verbose=False)
        dlearner.add_evidence(train_x, train_y)
        pred_y = dlearner.query(train_x)
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        insamrmse.append(in_rmse)
        pred_y = dlearner.query(test_x)
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        outsamrmse.append(out_rmse)


    plt.plot(leafs, insamrmse, marker='o', label='In-Sample RMSE')
    plt.plot(leafs, outsamrmse, marker='s', label='Out-Of-Sample RMSE')

    plt.title('Determinist Decision Tree Learner RMSE vs Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.xlim(50, 0)
    plt.xticks(np.arange(50, -1, -10))
    plt.grid()
    plt.legend()
    # plt.savefig('./experiment-1.png')
    # plt.show()
    plt.clf()

    #EXPERIMENT 2
    bagin = []
    bagout = []
    for l in leafs:
        blearner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':l}, bags=10, boost=False, verbose=False)
        blearner.add_evidence(train_x, train_y)
        pred_y = blearner.query(train_x)
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        bagin.append(in_rmse)
        pred_y = blearner.query(test_x)
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        bagout.append(out_rmse)

    plt.plot(leafs, insamrmse, marker='o', label='DT In-Sample RMSE')
    plt.plot(leafs, outsamrmse, marker='s', label='DT Out-Of-Sample RMSE')
    plt.plot(leafs, bagin, marker='^', label='Bag In-Sample RMSE')
    plt.plot(leafs, bagout, marker='v', label='Bag Out-Of-Sample RMSE')

    plt.title('Bagging of Determinist Decision Tree Learner RMSE vs Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.xlim(50, 0)
    plt.xticks(np.arange(50, -1, -10))
    plt.grid()
    plt.legend()
    # plt.savefig('./experiment-2.png')
    # plt.show()
    plt.clf()

    #EXPERIEMNT 3
    dtmae = []
    rtmae = []
    dtTime = []
    rtTime = []
    for l in leafs:
        start_time = time.time()
        dlearner = dt.DTLearner(leaf_size=l, verbose=False)
        dlearner.add_evidence(train_x, train_y)
        end_time = time.time()
        elapsed_time = end_time - start_time
        dtTime.append(elapsed_time)

        start_time = time.time()
        rlearner = dt.DTLearner(leaf_size=l, verbose=False)
        rlearner.add_evidence(train_x, train_y)
        end_time = time.time()
        elapsed_time = end_time - start_time
        rtTime.append(elapsed_time)


    # evaluate in sample
    # pred_y = dlearner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
    #
    # # evaluate out of sample
    # pred_y = dlearner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
