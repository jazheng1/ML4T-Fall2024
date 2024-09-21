import math
import sys

import numpy as np
from scipy import stats as st

class BagLearner:

    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        """
        This is a Bootstrap Aggregation Learner (BagLearner). You will need to properly implement this class as necessary.

        Parameters
            learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
            kwargs            - Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
            bags (int)        - The number of learners you should train using Bootstrap Aggregation.
                            If boost is true, then you should implement boosting (optional implementation).
            verbose (bool)    - If “verbose” is True, your code can print out information for debugging.
                            If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.models = []

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

        for b in range(0, self.bags):
            self.models.append(self.learner(**self.kwargs))
        for m in self.models:
            indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            sample_x = data_x[indices]
            sample_y = data_y[indices]
            m.add_evidence(sample_x, sample_y)

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
        for model in self.models:
            predicted_value = model.query(points)
            predictions.append(predicted_value)
        mean_predict = np.mean(predictions, axis=0)
        return mean_predict