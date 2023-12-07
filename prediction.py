import numpy as np

from sklearn.model_selection import train_test_split

from models.svm import run_model as run_svm
from models.random_forest import run_model as run_rf

class MLModelConfig:
    def __init__(self, X, y, X_train, X_test, y_train, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

def generate_sample_data():
    """
    Generate sample data
    """
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    return X, y

if __name__ == '__main__':
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create a configuration object
    model_config = MLModelConfig(X, y, X_train, X_test, y_train, y_test)

    # Running SVM prediction
    svm_results = run_svm(model_config)

    # Running Random Forest prediction
    rf_results = run_rf(model_config)

