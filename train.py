# This script can be used to train the final model version from scratch, which will save the final file in the pickled models folder.

# Library Import
import pandas as pd
from utils.Dataloader import PricingWizardDataset
from utils.DataTransformation import base_regression_pipeline, ridge_regression_pipeline
from utils.helpers import save_model, drop_helpers
from models import base_linear_regression, regularized_regression, regression_neural_network 
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Argument Parser
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", help="Name of model to train")
argParser.add_argument("-j", "--jobs", default=1 ,help="Number of jobs to run in parallel")

args = argParser.parse_args()

# Check if name is valid
if args.name not in ['base_regression', 'regularized_regression', 'neural_network']: # Add more models here when done
    print('Please enter a valid model name. Can be one of the following: base_regression, regularized_regression, neural_network')
    exit()
else:
    print('Training model:', args.name)

# Load data
data = PricingWizardDataset()


# Base Regression 
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


# Regularized Regression
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


# Neural Network
if args.name == 'neural_network':
    # Apply ridge regression data preparation
    print('Applying data preparation...')    
    data.apply_function(ridge_regression_pipeline)
    
    # Standard Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(drop_helpers(data.df))
    
    # Assigning X to data.df
    data.df[drop_helpers(data.df).columns] = X
    
    print('Done.')
    
    
    
    # Split data
    data.stratify_train_test_split(y_column='log_listing_price', 
                                   val_size=.2,
                                   return_splits=False)
    
    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(drop_helpers(data.X_train).to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(drop_helpers(data.X_test).to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(drop_helpers(data.X_val).to_numpy(), dtype=torch.float32)
    
    y_train_tensor = torch.tensor(data.y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(data.y_test, dtype=torch.float32)
    y_val_tensor = torch.tensor(data.y_val, dtype=torch.float32)
    
    # Create pytorch datasets
    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    valset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create pytorch dataloaders
    batch_size = 32 
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)
    
    # Train Model
    results = regression_neural_network.regression_network(train_loader, val_loader, X_test_tensor, data.y_test)
    print('R2 Score:', results['r2'])
    print('MSE:', results['mse'])
    print('MAE', results['mae'])
    print('RMSE', results['rmse'])

    # Save model
    path = 'models/pickled_models/regression_neural_net.pt'
    save_model(results, path, model_type='pytorch')

