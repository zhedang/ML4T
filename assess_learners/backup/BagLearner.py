import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def add_evidence(self, data_x, data_y):
        num_rows = data_x.shape[0]
        for learner in self.learners:
            # select random indices
            i = np.random.choice(num_rows, size=num_rows,replace=True)
            # select random entries
            learner.add_evidence(data_x[i], data_y[i])

    def author(self):
        return "zdang31"

    def study_group(self):
        return "zdang31"

    def query(self, points):
        results = np.mean([learner.query(points) for learner in self.learners], axis=0)
        return results

if __name__ == "__main__":

    Xtrain = np.random.rand(100, 5)  # 100 data points, 5 features
    Ytrain = np.random.rand(100)     # 100 target values

    # Generate separate random testing data
    Xtest = np.random.rand(30, 5)    # 30 data points, 5 features
    Ytest = np.random.rand(30)       # 30 target values

    # Instantiate BagLearner with DTLearner (as an example)
    bag_learner = BagLearner(
        learner=dt.DTLearner,
        kwargs={"leaf_size": 1},
        bags=20,
        boost=False,
        verbose=False
    )

    # Train the BagLearner
    bag_learner.add_evidence(Xtrain, Ytrain)

    # Query the BagLearner with the test set
    Ypred = bag_learner.query(Xtest)

    # Calculate RMSE manually using numpy
    mse = np.mean((Ytest - Ypred) ** 2)
    rmse = np.sqrt(mse)

    # Print out predictions and RMSE
    print("Predicted Y values for test data:")
    print(Ypred)
    print(f"RMSE on test data: {rmse:.4f}")