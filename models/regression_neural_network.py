# Import torch nn module
import torch
import torch.nn as nn
from utils.NeuralNetHelpers import train, train_step, test, set_device
from utils.RegressionEvaluation import regression_accuracy

# Neural network architecture
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.main(x)
    

def regression_network(train_loader, val_loader, test_tensor, test_target):
    """Function for training a a regression neural network"""
    
    # Instantiate model
    model = RegressionNN(next(iter(train_loader))[0].shape[1])
    
    # Determining device
    device = set_device()
        
    # Move model to device
    model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    results = train(model, 
                    criterion, 
                    optimizer, 
                    train_loader, 
                    val_loader, 
                    device, 
                    epochs=100, 
                    print_every=1, 
                    early_stopping=5)
    
    # Test model
    prediction = test(model, test_tensor, device)
    r2, mse, mae, rmse = regression_accuracy(prediction, test_target, return_metrics=True)

    # Create a dictionary to store results and model
    results = {
        'model': model,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }
    
    return results