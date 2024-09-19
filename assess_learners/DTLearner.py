import math
import sys

import numpy as np
from scipy import stats as st

class DTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        """
        This is a Decision Tree Learner (DTLearner). You will need to properly implement this class as necessary.

        Parameters
            leaf_size (int)  - Is the maximum number of samples to be aggregated at a leaf
            verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                                 If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        """
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        """
        Returns
            The GT username of the student

        Return type
            str
        """
        return "jzheng429"

    def study_group(self):
        """
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

        Return type
            str
        """
        return "jzheng429"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        """
        print("DATA: ", data_x, data_y)
        self.model = self.build_tree(data_x, data_y)
        print(self.model)

    def build_tree(self, data_x, data_y):
        # print(data_x, data_y)

        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            # fix
            leaf_val = data_y[0]
            return [-1, st.mode(leaf_val).mode[0], -1, -1]

        correlations = []
        for i in range(data_x.shape[1]):
            col = data_x[:, i]
            corr_coef = np.corrcoef(col, data_y)[0, 1]
            correlations.append(abs(corr_coef))  # not sure if ab is ok

        max_correlation = max(correlations)
        max_index = correlations.index(max_correlation)
        # print(correlations, max_correlation, max_index)

        SplitVal = np.median(data_x[:, max_index])
        # print('split: ', SplitVal)
        combined = np.column_stack((data_x, data_y))
        left_half = combined[combined[:, max_index] <= SplitVal]
        right_half = combined[combined[:, max_index] > SplitVal]

        lefttree = self.build_tree(left_half[:, :-1], left_half[:, -1])
        righttree = self.build_tree(right_half[:, :-1], right_half[:, -1])
        root = [max_index, SplitVal, 1, len(lefttree) + 1]
        return [root, lefttree, righttree]

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
        """
        return