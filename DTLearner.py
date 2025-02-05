import numpy as np
import math
class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree=None
    def author(self):
        return "zdang31"
    def study_group(self):
        return "zdang31"

    def build_tree(self,data):
        if data.shape[0] <= self.leaf_size:
            return np.atleast_2d(np.array(["leaf", np.mean(data[:, -1]), np.nan, np.nan],dtype=object))  # 2d fix the one row issue
        elif len(np.unique(data[:, -1])) == 1:
            return np.atleast_2d(np.array(["leaf", data[0, -1], np.nan, np.nan],dtype=object))
        else:
            # use if to prevent cases that whole column values are same
            corr = np.array([np.corrcoef(data[:, i], data[:, -1])[0, 1] if np.std(data[:, i]) > 0 else 0 for i in range(data.shape[1] - 1)])
            max_corr = np.where(np.isclose(np.abs(corr), np.max(np.abs(corr))))[0]  # isclose to fix index float point issue
            if len(max_corr) > 1:
                index = max_corr[0]
            else:
                index = np.argmax(np.abs(corr))
            split_value = np.median(data[:, index])
            # without check, recursion error
            left_data = data[data[:, index] <= split_value]
            right_data = data[data[:, index] > split_value]

            if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                return np.atleast_2d(np.array(["leaf", np.mean(data[:, -1]), np.nan, np.nan], dtype=object))
            left_tree = self.build_tree(left_data)
            right_tree = self.build_tree(right_data)
            root = np.array([index, split_value, 1, left_tree.shape[0] + 1],dtype=object)
            decision_tree = np.vstack((root, left_tree, right_tree))
            return decision_tree
        
    def add_evidence(self, data_x, data_y):
        data=np.hstack((data_x,data_y.reshape(-1, 1))) # make Ytrain 2d
        self.tree = self.build_tree(data)
    def query(self, points):
        arr=np.array([])
        for i in range(points.shape[0]):
            node=0
            while self.tree[node][0]!="leaf":
                if points[i][int(self.tree[node][0])]<=self.tree[node][1]:
                    node+=1
                else:
                    node+=self.tree[node][3]
                node=int(node)
            y=self.tree[node][1]
            arr=np.append(arr,y)
        return arr

# ✅ Run Model on Istanbul Data
if __name__ == "__main__":
    # Load CSV Data
    filename = "Data/Istanbul.csv"  # Adjust the path if needed
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)[:, 1:]  # Skip Date column

    # Split features (X) and target (Y)
    X = data[:, 1:-1]  # All columns except last
    Y = data[:, -1]   # Last column is the target

    # ✅ Split into Training (60%) & Testing (40%)
    np.random.seed(42)  # Ensures reproducibility
    indices = np.random.permutation(X.shape[0])
    train_size = int(0.6 * X.shape[0])

    train_x, test_x = X[indices[:train_size]], X[indices[train_size:]]
    train_y, test_y = Y[indices[:train_size]], Y[indices[train_size:]]

    # ✅ Train Decision Tree Learner
    learner = DTLearner(leaf_size=1, verbose=False)  # Try different leaf_size values
    learner.add_evidence(train_x, train_y)

    # ✅ Make Predictions
    train_pred = learner.query(train_x)
    test_pred = learner.query(test_x)

    # ✅ Helper Functions for Performance Metrics
    def rmse(y_true, y_pred):
        """Compute Root Mean Squared Error"""
        return math.sqrt(((y_true - y_pred) ** 2).mean())

    def correlation(y_true, y_pred):
        """Compute correlation coefficient"""
        return np.corrcoef(y_true, y_pred)[0, 1]

    # ✅ Evaluate Performance
    train_rmse = rmse(train_y, train_pred)
    test_rmse = rmse(test_y, test_pred)
    train_corr = correlation(train_y, train_pred)
    test_corr = correlation(test_y, test_pred)

    # ✅ Print Validation Results
    print("\n### DTLearner Performance on Istanbul Data ###")
    print(f"In-Sample RMSE: {train_rmse:.4f}")
    print(f"In-Sample Correlation: {train_corr:.4f}")
    print(f"Out-of-Sample RMSE: {test_rmse:.4f}")
    print(f"Out-of-Sample Correlation: {test_corr:.4f}")

    # ✅ Check for Issues
    assert not np.isnan(test_pred).any(), "Error: NaN values in predictions!"
    print("\n✅ Test Passed! DTLearner produces valid predictions.")