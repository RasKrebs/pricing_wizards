# Loading packages
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def regression_accuracy(prediction:np.array, true_labels:np.array, return_metrics:bool = False, scale_up:bool = False):
    """Function for printing regression metrics


    Args:
        prediction (np.array): Model predictions.
        true_labels (np.array): Ground truth labels.
        return_metrics (bool, optional): Whether to return metrics. Defaults to False.
        scale_up (bool, optional): Whether to scale up log scaled predictions and true labels. Defaults to False.
    Returns: If specified, returns metrics. Otherwise prints metrics.
        float: r2 score
        float: mean squared error
        float: mean absolute error
        float: root mean squared error
    """
    
    # Scale up if specified
    if scale_up:
        prediction = np.exp(prediction)
        true_labels = np.exp(true_labels)

    # Calculate metrics
    r2 = r2_score(true_labels, prediction)
    mse = mean_squared_error(true_labels, prediction)
    mae = mean_absolute_error(true_labels, prediction)
    rmse = np.sqrt(mean_squared_error(true_labels, prediction))

    # Return or print metrics
    if return_metrics:
        return r2, mse, mae, rmse
    else:
        print('R2 Score:', r2)
        print('MSE:', mse)
        print('MAE', mae)
        print('RMSE', rmse)

def threshold_accuracy(prediction:np.array, true_labels:np.array, p:int=0.05, return_metrics:bool = False, scale_up:bool = False):
    """Function for computing accuracy given a threshold

    Args:
        prediction (np.array): Model predictions.
        true_labels (np.array): Ground truth labels.
        p (float, optional): Percentage threshold. Defaults to 0.05.
        return_metrics (bool, optional): Whether to return metrics. Defaults to False.

    Returns: If specified, returns metrics. Otherwise prints metrics.
        float: Threshold accuracy
    """
    # Function to append wether or not the prediction is within a certain threshold
    output = []
    
    # Scale up if specified
    if scale_up:
        prediction = np.exp(prediction)
        true_labels = np.exp(true_labels)

    # Loop over all predictions
    for i in range(len(prediction)):
        if abs(prediction[i] - true_labels[i]) <= p * true_labels[i]:
            output.append(1)
        else:
            output.append(0)

    # Return or print accuracy
    if return_metrics:
        return sum(output) / len(output)
    else:
        print('Threshold Accuracy', sum(output) / len(output))

