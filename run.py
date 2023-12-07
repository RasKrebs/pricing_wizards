from preprocessing import run_preprocessing
from prediction import run_predictions
from statistics import run_statistics
from clustering import run_clustering

class MLModelConfig:
    def __init__(self, X, y, X_train, X_test, y_train, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

if __name__ == '__main__':
    run_preprocessing()

    run_statistics()

    run_clustering()

    model_config: object = example_method()

    run_predictions(model_config)
