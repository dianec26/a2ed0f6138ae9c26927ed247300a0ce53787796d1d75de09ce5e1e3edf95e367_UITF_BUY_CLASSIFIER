import sys
import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def train_model(train_data):
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )
    logger.add(
        "logs/data_preprocessing_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function} | {message}",
        level="DEBUG",
    )

    to_drop_final = ["original_label", "buy_reason", "buy_strength"]
    train = train_data.drop(to_drop_final, axis=1)
    
    X = train.drop("buy", axis=1)
    y = train["buy"]
    
    process_start = datetime.now()
    logger.info(
        f"Starting model training at {process_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    models = {
        'LogisticRegression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'params': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__l1_ratio': [0.1, 0.5, 0.9]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    }
    
    best_model = None
    best_score = -np.inf
    model_results = []
    
    logger.info("Starting hyperparameter tuning and model evaluation")
    
    for model_name, config in models.items():
        logger.info(f"Training {model_name}...")
        
        if model_name == 'LogisticRegression':
            param_grids = []
            
            param_grids.append({
                'classifier__C': config['params']['classifier__C'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            })
            
            param_grids.append({
                'classifier__C': config['params']['classifier__C'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['saga']
            })
            
            param_grids.append({
                'classifier__C': config['params']['classifier__C'],
                'classifier__penalty': ['elasticnet'],
                'classifier__solver': ['saga'],
                'classifier__l1_ratio': config['params']['classifier__l1_ratio']
            })
            
            best_grid_score = -np.inf
            best_grid_model = None
            
            for i, param_grid in enumerate(param_grids):
                grid_search = GridSearchCV(
                    config['model'], 
                    param_grid, 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X, y)
                
                if grid_search.best_score_ > best_grid_score:
                    best_grid_score = grid_search.best_score_
                    best_grid_model = grid_search
                    
                logger.debug(f"Grid {i+1} best score: {grid_search.best_score_:.4f}")
            
            current_model = best_grid_model
            
        else:
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            current_model = grid_search
        
        cv_scores = cross_val_score(
            current_model.best_estimator_, X, y, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        final_model = current_model.best_estimator_
        final_model.fit(X, y)
        
        y_pred = final_model.predict(X)
        y_pred_proba = final_model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        auc_score = roc_auc_score(y, y_pred_proba)
        
        model_results.append({
            'model': model_name,
            'best_params': current_model.best_params_,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'train_accuracy': accuracy,
            'train_auc': auc_score,
            'model_object': final_model
        })
        
        logger.info(f"{model_name} - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info(f"{model_name} - Train AUC: {auc_score:.4f}")
        logger.info(f"{model_name} - Best params: {current_model.best_params_}")
        
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = final_model
            best_model_name = model_name
            print("best model is ", model_name)
    
    process_end = datetime.now()
    logger.info(
        f"Finished model training at {process_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    feature_importance_df = get_feature_importance(best_model, X.columns, best_model_name)
    feature_importance_df.to_csv("logs/model_feature_importance.csv", index=False)
    logger.info("Saved feature importance")
    
    leaderboard_df = pd.DataFrame(model_results)
    leaderboard_df = leaderboard_df.sort_values('cv_auc_mean', ascending=False)
    leaderboard_df.drop('model_object', axis=1).to_csv("logs/model_leaderboard.csv", index=False)
    logger.info("Saved model leaderboard")
    
    with open(f"models/best_model_{best_model_name.lower()}.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    logger.info(f"Best model: {best_model_name} with CV AUC: {best_score:.4f}")
    logger.info("Returning best model")
    
    return best_model


def get_feature_importance(model, feature_names, model_name):
    if model_name == 'LogisticRegression':
        if hasattr(model, 'named_steps'):
            coefficients = model.named_steps['classifier'].coef_[0]
        else:
            coefficients = model.coef_[0]
        importance = np.abs(coefficients)
    
    elif model_name == 'XGBoost':
        importance = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df