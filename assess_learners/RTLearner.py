class RTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        """
        This is a This is a Random Tree Learner (RTLearner). You will need to properly implement this class as necessary.

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

    def add_evidence(data_x, data_y):
        """
        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        """
        return

    def query(points):
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