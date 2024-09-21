import numpy as np
import BagLearner as bag
import LinRegLearner as lrl
class InsaneLearner:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.models = []
    def author(self):
        return "jzheng429"
    def study_group(self):
        return "jzheng429"
    def add_evidence(self, data_x, data_y):
        for i in range(20): self.models.append(bag.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def query(self, points):
        predictions = []
        for model in self.models:
            predicted_value = model.query(points)
            predictions.append(predicted_value)
        predictions = np.array(predictions)
        return predictions