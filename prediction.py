import numpy as np

from sklearn.model_selection import train_test_split

from models.svm import run_model as run_svm
from models.random_forest import run_model as run_rf
from feature_importance import rank_feature_importances

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
    model_config: object = MLModelConfig(X, y, X_train, X_test, y_train, y_test)

    # Running SVM prediction
    svm_results: dict = run_svm(model_config)

    # Running Random Forest prediction
    rf_results: dict = run_rf(model_config)

    # Extract only feature importances from the results
    feature_importances: dict = {
        'SVR Linear': svm_results['SVR Linear']['feature_importances'],
        'Random Forest': rf_results['Random Forest']['feature_importances']
    }

    # Rank feature importances based on the values
    # ranked_feature_importances = rank_feature_importances(feature_importances)

