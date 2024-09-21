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
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        self.tree = []
        self._build_tree_helper(data_x, data_y)
        return self.tree

    def _build_tree_helper(self, data_x, data_y):
        if data_x.shape[0] == 0:
            return -1

        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            leaf_val = st.mode(data_y).mode[0]
            node_index = len(self.tree)
            self.tree.append([-1, leaf_val, -1, -1])
            return node_index

        correlations = []
        for i in range(data_x.shape[1]):
            col = data_x[:, i]
            corr_coef = np.corrcoef(col, data_y)[0, 1]
            if np.isnan(corr_coef):
                corr_coef = 0
            correlations.append(abs(corr_coef))

        max_correlation = max(correlations)
        best_feature = correlations.index(max_correlation)
        SplitVal = np.median(data_x[:, best_feature])


        # Check for identical split values
        if np.all(data_x[:, best_feature] == data_x[0, best_feature]):
            leaf_val = st.mode(data_y).mode[0]
            node_index = len(self.tree)
            self.tree.append([-1, leaf_val, -1, -1])
            return node_index

        left_indices = data_x[:, best_feature] <= SplitVal
        right_indices = data_x[:, best_feature] > SplitVal
        # print(max_correlation, best_feature, SplitVal)
        # print("L,R: ", left_indices, right_indices)

        if np.sum(right_indices) == 0:
            leaf_val = st.mode(data_y[left_indices]).mode[0]
            right_node_index = len(self.tree)
            self.tree.append([-1, leaf_val, -1, -1])
            return right_node_index
        else:
            right_node_index = self._build_tree_helper(data_x[right_indices], data_y[right_indices])

        if np.sum(left_indices) == 0:
            leaf_val = st.mode(data_y[right_indices]).mode[0]
            left_node_index = len(self.tree)
            self.tree.append([-1, leaf_val, -1, -1])
            return left_node_index
        else:
            left_node_index = self._build_tree_helper(data_x[left_indices], data_y[left_indices])

        node_index = len(self.tree)
        self.tree.append([best_feature, SplitVal, left_node_index, right_node_index])
        return node_index

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

        return predictions

    def query_tree(self, root, data_point):

        if root[0] == -1:
            return root[1]

        feature_index = root[0]
        split_value = root[1]
        if data_point[feature_index] <= split_value:
            return self.query_tree(self.tree[root[2]], data_point)
        else:
            return self.query_tree(self.tree[root[3]], data_point)