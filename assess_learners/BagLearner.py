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

    def add_evidence(self, data_x, data_y):
        #keep the number of rows in the bag data
        num_rows = data_x.shape[0]
        #loop through every learner;randomly choose indices of exact number of rows in bag data, with replacement
        for learner in self.learners:
            indices = np.random.choice(num_rows, size=num_rows,replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])

    def author(self):
        return "zdang31"

    def study_group(self):
        return "zdang31"

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)

if __name__ == "__main__":
    import numpy as np
    import RTLearner as rt  # Assuming RTLearner is implemented in a file named RTLearner_.py
    from BagLearner_ import BagLearner  # Ensure BagLearner is correctly imported

    # Load data from CSV file
    data = np.genfromtxt("Data/Istanbul.csv", delimiter=",")  # Assuming no header

    # Split into features (X) and target values (Y)
    X = data[1:, 1:-1]  # Exclude the first row and first column
    Y = data[1:, -1]  # Target values

    # Number of runs
    num_runs = 100

    # Store correlation results
    corrs_1_bag = []
    corrs_20_bags = []

    # Run the experiment 100 times
    for _ in range(num_runs):
        # Shuffle and split into train and test sets
        indices = np.random.permutation(len(X))
        train_size = int(0.6 * len(X))  # 60% training, 40% testing

        train_idx, test_idx = indices[:train_size], indices[train_size:]
        Xtrain, Xtest = X[train_idx], X[test_idx]
        Ytrain, Ytest = Y[train_idx], Y[test_idx]

        # Train BagLearner with 1 bag
        bag_learner_1 = BagLearner(
            learner=rt.RTLearner,
            kwargs={"leaf_size": 1},  # Adjust leaf_size if needed
            bags=1,  # Only one learner
            boost=False,
            verbose=False
        )
        bag_learner_1.add_evidence(Xtrain, Ytrain)
        Ytest_pred_1 = bag_learner_1.query(Xtest)

        # Train BagLearner with 20 bags
        bag_learner_20 = BagLearner(
            learner=rt.RTLearner,
            kwargs={"leaf_size": 1},  # Same leaf_size for fair comparison
            bags=20,  # 20 learners
            boost=False,
            verbose=False
        )
        bag_learner_20.add_evidence(Xtrain, Ytrain)
        Ytest_pred_20 = bag_learner_20.query(Xtest)

        # Compute out-of-sample correlation
        corr_1_bag = np.corrcoef(Ytest, Ytest_pred_1)[0, 1]
        corr_20_bags = np.corrcoef(Ytest, Ytest_pred_20)[0, 1]

        # Store results
        corrs_1_bag.append(corr_1_bag)
        corrs_20_bags.append(corr_20_bags)

    # Compute mean correlation
    mean_corr_1_bag = np.mean(corrs_1_bag)
    mean_corr_20_bags = np.mean(corrs_20_bags)

    # Print results
    print(f"Mean out-of-sample correlation with 1 bag over {num_runs} runs: {mean_corr_1_bag:.4f}")
    print(f"Mean out-of-sample correlation with 20 bags over {num_runs} runs: {mean_corr_20_bags:.4f}")

