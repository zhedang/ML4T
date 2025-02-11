import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.bags = bags
        self.verbose = verbose
        self.kwargs = kwargs
        self.learners = [learner(**kwargs) for i in range(bags)]
    def author(self):
        return "zdang31"
    def study_group(self):
        return "zdang31"

    def add_evidence(self, data_x, data_y):
        #keep the number of rows in the bag data
        num_rows = data_x.shape[0]
        #loop through every learner;randomly choose indices of exact number of rows in bag data, with replacement
        for learner in self.learners:
            indices = np.random.choice(num_rows, size=num_rows,replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)



