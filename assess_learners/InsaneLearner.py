import numpy as np 
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self,verbose=False):
        self.verbose=verbose
        self.learners=[bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False) for i in range(20)]
    def author(self):
        return "zdang31"
    def add_evidence(self,data_x,data_y):
        for learner in self.learners:
            learner.add_evidence(data_x,data_y)
    def query(self,points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)
