import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
import logging

class EdgeColoringRandomForest:
    """
    Random Forest model for edge coloring predictions, supporting both 
    classification (predicting edge colors) and regression (predicting coloring quality).
    """
    
    def __init__(self, mode='classification', n_estimators=100, max_depth=20, 
                 min_samples_leaf=2, class_weight='balanced', random_state=42):
        """
        Initialize the Random Forest model.
        
        Args:
            mode (str): 'classification' for color assignment or 'regression' for quality prediction
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            min_samples_leaf (int): Minimum samples required at a leaf node
            class_weight (str or dict): Weights for classes (classification only)
            random_state (int): Random seed for reproducibility
        """
        self.mode = mode
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        if mode == 'classification':
            self.params['class_weight'] = class_weight
            self.model = RandomForestClassifier(**self.params)
        else:  # regression
            self.model = RandomForestRegressor(**self.params)
        
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels/values
        
        Returns:
            self: The trained model
        """
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (numpy.ndarray): Feature matrix
        
        Returns:
            numpy.ndarray: Predicted labels/values
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        Only available for classification mode.
        
        Args:
            X (numpy.ndarray): Feature matrix
        
        Returns:
            numpy.ndarray: Probability estimates
        """
        if self.mode == 'classification':
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba is only available for classification mode")
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5, scoring=None):
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels/values
            param_grid (dict): Parameter grid to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
        
        Returns:
            dict: Best parameters found
        """
        if param_grid is None:
            if self.mode == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [10, 20, 30, 50],
                    'min_samples_leaf': [1, 2, 5, 10]
                }
            else:  # regression
                param_grid = {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [10, 20, 30, 50],
                    'min_samples_leaf': [1, 2, 5, 10]
                }
                
        if scoring is None:
            scoring = 'accuracy' if self.mode == 'classification' else 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.feature_importances_ = self.model.feature_importances_
        
        return grid_search.best_params_
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): Names of features
        
        Returns:
            dict or numpy.ndarray: Feature importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet")
            
        if feature_names is not None:
            return {feature: importance for feature, importance in 
                   zip(feature_names, self.feature_importances_)}
        return self.feature_importances_
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(filepath)
        self.feature_importances_ = self.model.feature_importances_
        return self