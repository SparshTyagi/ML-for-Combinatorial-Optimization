import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

def train_model(model, train_data, val_data, model_type='random_forest', 
                task='classification', epochs=100, lr=0.001, batch_size=32, 
                device='cuda', save_path=None, early_stopping=20, verbose=True):
    """
    Train a machine learning model for edge coloring.
    
    Args:
        model: The model to train
        train_data: Training data (format depends on model_type)
        val_data: Validation data (format depends on model_type)
        model_type: 'random_forest', 'gnn', or 'hybrid'
        task: 'classification' for edge coloring or 'regression' for quality prediction
        epochs: Number of epochs (only for neural network models)
        lr: Learning rate (only for neural network models)
        batch_size: Batch size (only for neural network models)
        device: Device to use for training ('cuda' or 'cpu')
        save_path: Path to save the best model
        early_stopping: Number of epochs to wait for improvement before stopping
        verbose: Whether to print progress
    
    Returns:
        Trained model, training history
    """
    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Random Forest training
    if model_type == 'random_forest':
        if task == 'classification':
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_preds = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_preds)
            
            # Calculate validation metrics
            val_preds = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            
            history['train_acc'] = [train_acc]
            history['val_acc'] = [val_acc]
            
            if verbose:
                print(f"Training accuracy: {train_acc:.4f}")
                print(f"Validation accuracy: {val_acc:.4f}")
                print(f"Training time: {time.time() - start_time:.2f} seconds")
        
        elif task == 'regression':
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_preds = model.predict(X_train)
            train_mse = mean_squared_error(y_train, train_preds)
            
            # Calculate validation metrics
            val_preds = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            
            history['train_loss'] = [train_mse]
            history['val_loss'] = [val_mse]
            
            if verbose:
                print(f"Training MSE: {train_mse:.4f}")
                print(f"Validation MSE: {val_mse:.4f}")
                print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # GNN or hybrid model training
    elif model_type in ['gnn', 'hybrid']:
        # Set up device
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        model = model.to(device)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        if task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            criterion = nn.MSELoss()
        
        # Set up data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", 
                                disable=not verbose)
            
            for batch in train_iterator:
                # Move batch to device
                batch = batch.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                if task == 'classification':
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    
                    # Calculate accuracy
                    pred = out.argmax(dim=1)
                    train_correct += (pred == batch.y).sum().item()
                    train_total += batch.y.size(0)
                else:  # regression
                    out = model(batch)
                    loss = criterion(out, batch.y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                
                # Update progress bar
                if task == 'classification':
                    train_iterator.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        'acc': f"{train_correct/max(1, train_total):.4f}"
                    })
                else:
                    train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            if task == 'classification':
                train_acc = train_correct / train_total
                history['train_acc'].append(train_acc)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", 
                                   disable=not verbose)
                
                for batch in val_iterator:
                    batch = batch.to(device)
                    
                    if task == 'classification':
                        out = model(batch)
                        loss = criterion(out, batch.y)
                        
                        # Calculate accuracy
                        pred = out.argmax(dim=1)
                        val_correct += (pred == batch.y).sum().item()
                        val_total += batch.y.size(0)
                    else:  # regression
                        out = model(batch)
                        loss = criterion(out, batch.y)
                    
                    val_loss += loss.item() * batch.num_graphs
                    
                    # Update progress bar
                    if task == 'classification':
                        val_iterator.set_postfix({
                            'loss': f"{loss.item():.4f}", 
                            'acc': f"{val_correct/max(1, val_total):.4f}"
                        })
                    else:
                        val_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            if task == 'classification':
                val_acc = val_correct / val_total
                history['val_acc'].append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict()
                
                # Save the best model
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print epoch summary
            if verbose:
                if task == 'classification':
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
                          f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Load the best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        if verbose:
            print(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    return model, history

def evaluate_model(model, test_data, model_type='random_forest', task='classification', 
                  batch_size=32, device='cuda', verbose=True):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_data: Test data (format depends on model_type)
        model_type: 'random_forest', 'gnn', or 'hybrid'
        task: 'classification' for edge coloring or 'regression' for quality prediction
        batch_size: Batch size (only for neural network models)
        device: Device to use for evaluation ('cuda' or 'cpu')
        verbose: Whether to print progress
    
    Returns:
        Dictionary of evaluation metrics
    """
    start_time = time.time()
    metrics = {}
    
    # Random Forest evaluation
    if model_type == 'random_forest':
        if task == 'classification':
            X_test, y_test = test_data
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics['accuracy'] = accuracy
            metrics['classification_report'] = report
            
            if verbose:
                print(f"Test accuracy: {accuracy:.4f}")
                print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
                print(classification_report(y_test, y_pred))
        
        elif task == 'regression':
            X_test, y_test = test_data
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            
            metrics['mse'] = mse
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            
            if verbose:
                print(f"Test MSE: {mse:.4f}")
                print(f"Test RMSE: {rmse:.4f}")
                print(f"Test MAE: {mae:.4f}")
                print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
    
    # GNN or hybrid model evaluation
    elif model_type in ['gnn', 'hybrid']:
        # Set up device
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Set up data loader
        test_loader = DataLoader(test_data, batch_size=batch_size)
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        if task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            criterion = nn.MSELoss()
        
        with torch.no_grad():
            test_iterator = tqdm(test_loader, desc="Evaluation", disable=not verbose)
            
            for batch in test_iterator:
                batch = batch.to(device)
                
                if task == 'classification':
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    
                    # Calculate accuracy
                    pred = out.argmax(dim=1)
                    test_correct += (pred == batch.y).sum().item()
                    test_total += batch.y.size(0)
                    
                    # Save predictions and labels for detailed metrics
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
                else:  # regression
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    
                    # Save predictions and labels for detailed metrics
                    all_preds.append(out.cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
                
                test_loss += loss.item() * batch.num_graphs
        
        test_loss /= len(test_loader.dataset)
        metrics['test_loss'] = test_loss
        
        # Combine predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        if task == 'classification':
            test_accuracy = test_correct / test_total
            metrics['accuracy'] = test_accuracy
            
            # Detailed classification metrics
            report = classification_report(all_labels, all_preds, output_dict=True)
            metrics['classification_report'] = report
            
            if verbose:
                print(f"Test loss: {test_loss:.4f}")
                print(f"Test accuracy: {test_accuracy:.4f}")
                print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
                print(classification_report(all_labels, all_preds))
        else:  # regression
            mse = mean_squared_error(all_labels, all_preds)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_labels - all_preds))
            
            metrics['mse'] = mse
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            
            if verbose:
                print(f"Test loss: {test_loss:.4f}")
                print(f"Test MSE: {mse:.4f}")
                print(f"Test RMSE: {rmse:.4f}")
                print(f"Test MAE: {mae:.4f}")
                print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
    
    return metrics

def cross_validate(model_class, data, n_splits=5, model_type='random_forest', task='classification', 
                  **kwargs):
    """
    Perform cross-validation for model evaluation.
    
    Args:
        model_class: Class of the model to train
        data: Complete dataset (format depends on model_type)
        n_splits: Number of cross-validation folds
        model_type: 'random_forest', 'gnn', or 'hybrid'
        task: 'classification' for edge coloring or 'regression' for quality prediction
        **kwargs: Additional arguments for model training
    
    Returns:
        List of metrics for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    if model_type == 'random_forest':
        X, y = data
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Further split training data into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)
            
            # Initialize and train model
            model = model_class(**kwargs)
            model, _ = train_model(
                model, (X_train, y_train), (X_val, y_val), 
                model_type=model_type, task=task, verbose=True
            )
            
            # Evaluate model
            metrics = evaluate_model(
                model, (X_test, y_test), 
                model_type=model_type, task=task, verbose=True
            )
            
            fold_metrics.append(metrics)
            print("-" * 50)
    
    # For GNN models, cross-validation would be implemented differently
    # due to the nature of graph data (typically split at the graph level)
    # This would be more complex and is simplified here
    elif model_type in ['gnn', 'hybrid']:
        raise NotImplementedError(
            "Cross-validation for GNN models requires custom implementation based on specific dataset structure."
        )
    
    return fold_metrics