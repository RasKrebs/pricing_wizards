from utils.RegressionEvaluation import regression_accuracy
from sklearn.linear_model import LinearRegression
from utils.helpers import drop_helpers

# Model
def linear_regression(dataset):
    """
    Run a machine learning model using hyperparameter tuning with both GridSearchCV and RandomizedSearchCV.

    Parameters:
    - dataset (object): An object containing data and configurations for model training and testing.

    Returns:
    - results (dict): A dictionary containing the regression results and model
    """

    # Create a Linear Regression model
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(drop_helpers(dataset.X_train), dataset.y_train)

    # Make predictions on the test data
    y_pred = lr.predict(drop_helpers(dataset.X_test))

    # Calculate metrics
    r2, mse, mae, rmse = regression_accuracy(y_pred, dataset.y_test, return_metrics=True)

    # Create a dictionary to store results and model
    results = {
        'model': lr,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }

    return results