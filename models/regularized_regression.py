from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from utils.RegressionEvaluation import regression_accuracy
from utils.helpers import drop_helpers
import numpy as np

# Model
def regularized_regression(dataset, n_jobs=1):
    """
    Run a machine learning model using hyperparameter tuning with both GridSearchCV and RandomizedSearchCV.

    Parameters:
    - dataset (object): An object containing data and configurations for model training and testing.

    Returns:
    - results (dict): A dictionary containing the regression results and model
    """
    print('Training model using GridSearchCV: regularized_regression')
    
    # Ridge regression params
    param_grid = {'alpha': np.logspace(-3, 3, 13)}

    # Instantiate model
    model = Ridge() 

    # Grid search
    grid_search = GridSearchCV(model, 
                               param_grid, 
                               cv=5, 
                               scoring='neg_mean_squared_error', 
                               return_train_score=True, 
                               verbose=2, 
                               n_jobs = int(n_jobs))
    
    # Fit model
    grid_search.fit(drop_helpers(dataset.X_train), dataset.y_train)
    
    # Extract best model
    model = grid_search.best_estimator_
    
    # Test predictions
    y_pred = model.predict(drop_helpers(dataset.X_test))
    r2, mse, mae, rmse = regression_accuracy(y_pred, dataset.y_test, return_metrics=True)

    # Create a dictionary to store results and model
    results = {
        'model': model,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }

    return results