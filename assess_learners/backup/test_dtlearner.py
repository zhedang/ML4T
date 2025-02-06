import numpy as np
import DTLearner as dt  # Import your DTLearner class
import math

def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error"""
    return math.sqrt(((y_true - y_pred) ** 2).mean())

def correlation(y_true, y_pred):
    """Compute correlation coefficient"""
    return np.corrcoef(y_true, y_pred)[0, 1]

if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)  # Ensures reproducibility
    num_samples = 100
    num_features = 5

    Xtrain = np.random.rand(num_samples, num_features)  # Random features
    Ytrain = np.sum(Xtrain, axis=1) + np.random.randn(num_samples) * 0.1  # Y = Sum(X) + noise

    Xtest = np.random.rand(num_samples // 2, num_features)  # Test features
    Ytest = np.sum(Xtest, axis=1) + np.random.randn(num_samples // 2) * 0.1  # Test labels

    # Train Decision Tree Learner
    learner = dt.DTLearner(leaf_size=1, verbose=False)
    learner.add_evidence(Xtrain, Ytrain)

    # Make predictions
    Ypred_train = learner.query(Xtrain)
    Ypred_test = learner.query(Xtest)

    # Evaluate performance
    train_rmse = rmse(Ytrain, Ypred_train)
    test_rmse = rmse(Ytest, Ypred_test)
    train_corr = correlation(Ytrain, Ypred_train)
    test_corr = correlation(Ytest, Ypred_test)

    print("\n### DTLearner Performance ###")
    print(f"In-Sample RMSE: {train_rmse:.4f}")
    print(f"In-Sample Correlation: {train_corr:.4f}")
    print(f"Out-of-Sample RMSE: {test_rmse:.4f}")
    print(f"Out-of-Sample Correlation: {test_corr:.4f}")

    # Basic check if predictions make sense
    assert not np.isnan(Ypred_test).any(), "Error: NaN values in predictions!"
    print("\n✅ Test Passed! DTLearner produces valid predictions.")
