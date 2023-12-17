import torch
import numpy as np

def train_step(model, criterion, optimizer, train_loader, device):
    """
    Trains the model for one epoch.
    """
    # Set model to train mode
    model.train()
    
    # Save loss
    losses = []
    
    # Iterate over train_loader
    for i, data in enumerate(train_loader):
        
        # Extracting data and labels + moving to device
        input, labels = data
        input = input.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_hat = model(input).squeeze()
        
        # Compute loss
        loss = criterion(y_hat, labels)
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Save loss
        losses.append(loss.item())
    
    # Return loss
    return losses


def validation(model, validation_loader, criterion, device):
    """
    Performs a validation step.
    """
    # Set model to eval mode
    model.eval()
    
    # Save loss
    losses = []
    
    # Iterate over train_loader
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            # Extracting data and labels + moving to device
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            y_hat = model(inputs).squeeze()

            # Compute loss
            loss = criterion(y_hat, targets)
            
            # Save loss
            losses.append(loss.item())
    
    # Return loss
    return losses

def train(model, criterion, optimizer, train_loader, validation_loader, device, epochs=100, print_every=1, early_stopping=5):
    # Train method
    train_losses = []
    val_losses = []
    
    train_accuracies = {
        'r2': [],
        'mse': [],
        'mae': [],
        'rmse': []
    }
    
    # Early stopping
    epochs_no_improve = 0
    min_val_loss = np.Inf
    
    for epoch in range(epochs):
        # Train step
        train_loss = train_step(model, criterion, optimizer, train_loader, device)
        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)
        
        # Validation step
        val_loss = validation(model, validation_loader, criterion, device)
        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)
        
        # Print loss
        if epoch % print_every == 0:
            print(f'Epoch {epoch}, train loss: {train_loss}, val loss: {val_loss}')
        
        # Early stopping
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == early_stopping:
            print('Early stopping!')
            break
    
    return train_losses, val_losses

def test(model, test_tensors, device):
    """
    Performs a test step.
    """
    # Set model to eval mode
    model.eval()
    
    # Save loss
    losses = []
    
    # Test model
    with torch.no_grad():
        y_pred = model(test_tensors.to(device))
    
    # Return loss
    return y_pred.cpu().numpy().squeeze()