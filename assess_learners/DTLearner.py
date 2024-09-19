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
        self.tree = []

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
        self.tree = self.build_tree(data_x, data_y, 1, np.array([]))

        print('tree:', self.tree)

    def build_tree(self, data_x, data_y,l_index, tree):
        # print(tree)
        if data_x.shape[0] < self.leaf_size or len(np.unique(data_y)) == 1:
            # print('hi')
            leaf_val = st.mode(data_y).mode[0]  # Use the mode for leaf values
            tree = np.append(tree, [-1, leaf_val, -1, -1], axis = 0)
            return tree

        correlations = []
        for i in range(data_x.shape[1]):
            col = data_x[:, i]
            corr_coef = np.corrcoef(col, data_y)[0, 1]
            correlations.append(abs(corr_coef))

        max_correlation = max(correlations)
        best_feature = correlations.index(max_correlation)
        SplitVal = np.median(data_x[:, best_feature])
        combined = np.column_stack((data_x, data_y))
        left_half = combined[combined[:, best_feature] <= SplitVal]
        right_half = combined[combined[:, best_feature] > SplitVal]

        lefttree = self.build_tree(left_half[:, :-1], left_half[:, -1], l_index+1, np.array([]))
        righttree = self.build_tree(right_half[:, :-1], right_half[:, -1], lefttree.shape[0] + 1, np.array([]))

        root = [best_feature, SplitVal, l_index, lefttree.shape[0] + 1]
        print(tree)
        tree = np.append(tree, root, axis= 0)
        print('After: ', tree)
        return tree

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
        predictions = []
        for data_point in points:
            predicted_value = self.query_tree(self.tree[-1], data_point)
            predictions.append(predicted_value)

        # Convert predictions to a NumPy array if needed
        predictions = np.array(predictions)
        # print("Predictions: ", predictions)
        return predictions

    def query_tree(self, root, data_point):

        if root[0] == -1:
            return root[1]

        # Get the feature index and split value
        feature_index = root[0]
        split_value = root[1]

        # Traverse left or right subtree based on the data point
        if data_point[feature_index] <= split_value:
            return self.query_tree(self.tree[root[2]], data_point)  # Go to left subtree
        else:
            return self.query_tree(self.tree[root[3]], data_point)  # Go to right subtree