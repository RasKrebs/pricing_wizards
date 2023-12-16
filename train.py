# Library Import
import pandas as pd
from utils.Dataloader import PricingWizardDataset
from utils.DataTransformation import base_regression_pipeline, ridge_regression_pipeline
from utils.helpers import save_model
from models import base_linear_regression, regularized_regression
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", help="Name of model to train")
argParser.add_argument("-j", "--jobs", default=1 ,help="Number of jobs to run in parallel")

args = argParser.parse_args()
print('Training model:', args.name)

# Load data
data = PricingWizardDataset()


#### BASE_LINEAR_REGRESSION ####

if args.name == 'base_regression':
    # Apply data preparation
    print('Applying data preparation...')    
    data.apply_function(base_regression_pipeline)
    print('Done.')
    
    # Split data
    data.stratify_train_test_split(y_column='log_listing_price', 
                                   val_size=0,
                                   return_splits=False)

    # Run model
    results = base_linear_regression.linear_regression(data)
    print('R2 Score:', results['r2'])
    print('MSE:', results['mse'])
    print('MAE', results['mae'])
    print('RMSE', results['rmse'])

    # Save model
    path = 'models/pickled_models/base_regression.pkl'
    save_model(results, path)

data.reset_dataset()

if args.name == 'regularized_regression':
    # Apply data preparation
    print('Applying data preparation...')    
    data.apply_function(ridge_regression_pipeline)
    print('Done.')
    
    # Split data
    data.stratify_train_test_split(y_column='log_listing_price', 
                                   val_size=0,
                                   return_splits=False)
    
    # Make predictions on the test data
    results = regularized_regression.regularized_regression(data, args.jobs)

    print('R2 Score:', results['r2'])
    print('MSE:', results['mse'])
    print('MAE', results['mae'])
    print('RMSE', results['rmse'])

    # Save model
    path = 'models/pickled_models/regularized_regression.pkl'
    save_model(results, path)

#### New Model ####