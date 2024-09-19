import math
import sys

import numpy as np
from scipy import stats as st

class RTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        """
        This is a Random Tree Learner (RTLearner). You will need to properly implement this class as necessary.

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
        self.build_tree(data_x, data_y)
        print("model:", self.tree)

    def build_tree(self, data_x, data_y):
        print(data_x)
        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            # print(data_y)
            leaf_val = st.mode(data_y).mode[0]
            node_index = len(self.tree)
            self.tree.append([-1, leaf_val, -1, -1])
            return node_index

        random_feature = np.random.randint(0, data_x.shape[1])
        SplitVal = np.median(data_x[:, random_feature])
        combined = np.column_stack((data_x, data_y))
        left_half = combined[combined[:, random_feature] <= SplitVal]
        right_half = combined[combined[:, random_feature] > SplitVal]

        left_index = self.build_tree(left_half[:, :-1], left_half[:, -1])
        right_index = self.build_tree(right_half[:, :-1], right_half[:, -1])

        node_index = len(self.tree)
        self.tree.append([random_feature, SplitVal, left_index, right_index])

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