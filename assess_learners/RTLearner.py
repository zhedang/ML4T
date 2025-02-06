import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "zdang31"

    def study_group(self):
        return "zdang31"

    def add_evidence(self, data_x, data_y):
        # end recursion when reaching leaf size
        if data_x.shape[0] <= self.leaf_size:
            # 2d fix shape issue. otherwise, (3,) instead of (3,1) will cause mistake for right tree calculation
            tree = np.atleast_2d(np.array(["leaf", np.mean(data_y), np.nan, np.nan], dtype=object))
        # end recursion when all values are same
        elif len(np.unique(data_y)) == 1:
            tree = np.atleast_2d(np.array(["leaf", data_y[0], np.nan, np.nan], dtype=object))
        else:
            # randomly select a feature
            index= np.random.randint(0,data_x.shape[1])
            # using median of selected column as split value.
            split_value = np.median(data_x[:, index])
            # slicing left and right tree data
            left_indices = data_x[:, index] <= split_value
            right_indices = data_x[:, index] > split_value
            left_data_x = data_x[left_indices]
            left_data_y = data_y[left_indices]
            right_data_x = data_x[right_indices]
            right_data_y = data_y[right_indices]
            # prevent edge case for example [1,2,2].
            if left_data_x.shape[0] == 0 or right_data_x.shape[0] == 0:
                tree = np.atleast_2d(np.array(["leaf", np.mean(data_y), np.nan, np.nan], dtype=object))
            else:
                left_tree = self.add_evidence(left_data_x, left_data_y)
                right_tree = self.add_evidence(right_data_x, right_data_y)
                root = np.array([index, split_value, 1, left_tree.shape[0] + 1], dtype=object)
                tree = np.vstack((root, left_tree, right_tree))
        self.tree = tree
        return tree

    def query(self, points):
        arr = np.array([])
        for i in range(points.shape[0]):
            node = 0
            while self.tree[node][0] != "leaf":
                if points[i][int(self.tree[node][0])] <= self.tree[node][1]:
                    node += 1
                else:
                    node += self.tree[node][3]
                node = int(node)
            y = self.tree[node][1]
            arr = np.append(arr, y)
        return arr


# ✅ Run Model on Istanbul Data
import numpy as np
import RTLearner as rt

if __name__ == "__main__":
    # Load CSV Data
    filename = "Data/winequality-white.csv"  # Adjust the path if needed
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)[:, :]  # Skip Date column

    # Set seed for reproducibility
    np.random.seed(1481090001)

    # Shuffle data indices
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]

    # Split data into train (60%) and test (40%)
    cutoff = int(0.6 * data.shape[0])
    train_x, test_x = data[:cutoff, :-1], data[cutoff:, :-1]  # Features
    train_y, test_y = data[:cutoff, -1], data[cutoff:, -1]  # Target

    # Train RTLearner
    learner = rt.RTLearner(leaf_size=1, verbose=False)
    learner.add_evidence(train_x, train_y)

    # In-sample prediction
    pred_y_train = learner.query(train_x)

    # Compute correlation
    correlation = np.corrcoef(pred_y_train, train_y)[0, 1]

    # Output result
    print(f"In-sample correlation with leaf_size=1: {correlation:.6f}")

    # Validate against the required threshold
    if correlation >= 0.95:
        print("Test Passed: In-sample correlation meets the requirement.")
    else:
        print("Test Failed: In-sample correlation is below the required 0.95 threshold.")

