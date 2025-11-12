"""
Hyperparameter tuning module for SpliceRead with nested cross-validation.

This module provides comprehensive hyperparameter optimization using nested cross-validation
to avoid data leakage and ensure robust model selection.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from .training_utils import (
    create_directories, calculate_class_distribution, calculate_class_weights,
    to_categorical, deep_cnn_classifier, save_training_log, evaluate_fold,
    save_fold_results, summarize_cross_validation
)
from models.cnn_classifier import create_model_with_hyperparameters
# Note: No direct synthetic generation function is needed/used here.

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning with nested cross-validation.
    """
    
    def __init__(self, output_dir: str = './hyperparameter_tuning_results'):
        self.output_dir = output_dir
        create_directories(output_dir)
        
        # Define hyperparameter search spaces
        self.param_spaces = self._define_parameter_spaces()
        
        # Results storage
        self.tuning_results = []
        self.best_params = None
        self.best_score = -np.inf
        
    def _define_parameter_spaces(self) -> Dict[str, Dict]:
        """
        Define hyperparameter search spaces for different optimization methods.
        """
        return {
            'grid_search': {
                # Model architecture
                'num_conv_layers': [2, 3, 4],
                'num_filters': [32, 64, 128],
                'kernel_size': [3, 5, 7],
                'dropout_rate': [0.1, 0.2, 0.3, 0.5],
                
                # Training parameters
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [16, 32, 64],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                
                # Data augmentation (fixed synthetic ratio, tune method only)
                'augmentation_method': ['adasyn', 'svm_adasyn'],
                
                # Regularization
                'l2_reg': [0.0, 0.001, 0.01],
                'early_stopping_patience': [10, 20, 30]
            },
            
            'random_search': {
                # Model architecture (continuous ranges)
                'num_conv_layers': [2, 3, 4, 5],
                'num_filters': [16, 32, 64, 128, 256],
                'kernel_size': [3, 5, 7, 9],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                
                # Training parameters
                'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                'batch_size': [8, 16, 32, 64, 128],
                'optimizer': ['adam', 'sgd', 'rmsprop', 'adamax'],
                
                # Data augmentation (fixed synthetic ratio, tune method only)
                'augmentation_method': ['adasyn', 'svm_adasyn'],
                
                # Regularization
                'l2_reg': [0.0, 0.0001, 0.001, 0.01, 0.1],
                'early_stopping_patience': [5, 10, 15, 20, 25, 30]
            },
            
            'bayesian_search': {
                # Continuous parameters for Bayesian optimization
                'learning_rate': (0.0001, 0.1),
                'dropout_rate': (0.1, 0.5),
                'l2_reg': (0.0, 0.01),
                
                # Discrete parameters
                'num_conv_layers': [2, 3, 4, 5],
                'num_filters': [32, 64, 128, 256],
                'kernel_size': [3, 5, 7],
                'batch_size': [16, 32, 64, 128],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'augmentation_method': ['adasyn', 'svm_adasyn'],
                'early_stopping_patience': [10, 20, 30]
            }
        }
    
    def nested_cross_validation_with_hyperparameter_tuning(
        self,
        acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
        outer_folds: int = 5,
        inner_folds: int = 3,
        optimization_method: str = 'random_search',
        n_trials: int = 50,
        epochs: int = 50,
        sequence_length: int = 600,
        synthetic_ratio: float = 100.0,
        augmentation_method: str = 'adasyn'
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation with hyperparameter tuning.
        
        Args:
            acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data: Separated data
            outer_folds: Number of outer CV folds for final evaluation
            inner_folds: Number of inner CV folds for hyperparameter tuning
            optimization_method: 'grid_search', 'random_search', or 'bayesian_search'
            n_trials: Number of trials for random/bayesian search
            epochs: Number of training epochs
            sequence_length: Sequence length in bp
            synthetic_ratio: Fixed synthetic ratio (not tuned)
            augmentation_method: Fixed augmentation method (not tuned)
            
        Returns:
            Dictionary containing best parameters and results
        """
        
        print(f"\n{'='*80}")
        print(f"NESTED CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
        print(f"{'='*80}")
        print(f"Outer folds: {outer_folds}")
        print(f"Inner folds: {inner_folds}")
        print(f"Optimization method: {optimization_method.upper()}")
        print(f"Number of trials: {n_trials}")
        print(f"Training epochs: {epochs}")
        print(f"Sequence length: {sequence_length}bp")
        print(f"Fixed synthetic ratio: {synthetic_ratio}%")
        print(f"Fixed augmentation method: {augmentation_method.upper()}")
        print(f"{'='*80}\n")
        
        # Combine all data for outer CV
        all_data = np.concatenate([acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data])
        all_labels = np.concatenate([
            np.zeros(len(acc_can_data)),      # Acceptor canonical: 0
            np.zeros(len(acc_nc_data)),       # Acceptor non-canonical: 0  
            np.ones(len(don_can_data)),       # Donor canonical: 1
            np.ones(len(don_can_data)),       # Donor non-canonical: 1
            np.full(len(neg_data), 2)         # Negative: 2
        ])
        
        # Create indices for data separation
        acc_can_indices = list(range(0, len(acc_can_data)))
        acc_nc_indices = list(range(len(acc_can_data), len(acc_can_data) + len(acc_nc_data)))
        don_can_indices = list(range(len(acc_can_data) + len(acc_nc_data), 
                                    len(acc_can_data) + len(acc_nc_data) + len(don_can_data)))
        don_nc_indices = list(range(len(acc_can_data) + len(acc_nc_data) + len(don_can_data),
                                   len(acc_can_data) + len(acc_nc_data) + len(don_can_data) + len(don_nc_data)))
        neg_indices = list(range(len(acc_can_data) + len(acc_nc_data) + len(don_can_data) + len(don_nc_data),
                                len(all_data)))
        
        # Outer cross-validation
        from sklearn.model_selection import KFold
        outer_kfold = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
        
        outer_results = []
        best_params_per_fold = []
        
        for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_kfold.split(all_data)):
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {outer_fold + 1}/{outer_folds}")
            print(f"{'='*60}")
            
            # Split data for outer fold
            outer_train_data = all_data[outer_train_idx]
            outer_train_labels = all_labels[outer_train_idx]
            outer_val_data = all_data[outer_val_idx]
            outer_val_labels = all_labels[outer_val_idx]
            
            # Separate outer training data by type
            outer_train_acc_can_idx = [i for i in outer_train_idx if i in acc_can_indices]
            outer_train_acc_nc_idx = [i for i in outer_train_idx if i in acc_nc_indices]
            outer_train_don_can_idx = [i for i in outer_train_idx if i in don_can_indices]
            outer_train_don_nc_idx = [i for i in outer_train_idx if i in don_nc_indices]
            outer_train_neg_idx = [i for i in outer_train_idx if i in neg_indices]
            
            # Get separated data for outer training
            outer_acc_can_data = all_data[outer_train_acc_can_idx] if outer_train_acc_can_idx else np.array([])
            outer_acc_nc_data = all_data[outer_train_acc_nc_idx] if outer_train_acc_nc_idx else np.array([])
            outer_don_can_data = all_data[outer_train_don_can_idx] if outer_train_don_can_idx else np.array([])
            outer_don_nc_data = all_data[outer_train_don_nc_idx] if outer_train_don_nc_idx else np.array([])
            outer_neg_data = all_data[outer_train_neg_idx] if outer_train_neg_idx else np.array([])
            
            print(f"[INFO] Outer fold {outer_fold + 1} training data:")
            print(f"  Acceptor canonical: {len(outer_acc_can_data)}")
            print(f"  Acceptor non-canonical: {len(outer_acc_nc_data)}")
            print(f"  Donor canonical: {len(outer_don_can_data)}")
            print(f"  Donor non-canonical: {len(outer_don_nc_data)}")
            print(f"  Negative: {len(outer_neg_data)}")
            print(f"  Validation: {len(outer_val_data)}")
            
            # Hyperparameter tuning on outer training data
            print(f"\n[INFO] Starting hyperparameter tuning for outer fold {outer_fold + 1}...")
            best_params = self._tune_hyperparameters(
                outer_acc_can_data, outer_acc_nc_data, outer_don_can_data, 
                outer_don_nc_data, outer_neg_data,
                inner_folds=inner_folds,
                optimization_method=optimization_method,
                n_trials=n_trials,
                epochs=epochs,
                sequence_length=sequence_length,
                outer_fold=outer_fold + 1,
                synthetic_ratio=synthetic_ratio,
                augmentation_method=augmentation_method
            )
            
            best_params_per_fold.append(best_params)
            
            # Train final model on outer training data with best parameters
            print(f"\n[INFO] Training final model for outer fold {outer_fold + 1} with best parameters...")
            final_model = self._train_final_model(
                outer_acc_can_data, outer_acc_nc_data, outer_don_can_data,
                outer_don_nc_data, outer_neg_data,
                best_params, epochs, sequence_length, synthetic_ratio, augmentation_method
            )
            
            # Evaluate on outer validation set
            print(f"[INFO] Evaluating final model on outer validation set...")
            val_predictions = final_model.predict(outer_val_data)
            val_pred_classes = np.argmax(val_predictions, axis=1)
            
            val_accuracy = accuracy_score(outer_val_labels, val_pred_classes)
            val_f1 = f1_score(outer_val_labels, val_pred_classes, average='weighted')
            val_precision = precision_score(outer_val_labels, val_pred_classes, average='weighted')
            val_recall = recall_score(outer_val_labels, val_pred_classes, average='weighted')
            
            outer_results.append({
                'outer_fold': outer_fold + 1,
                'best_params': best_params,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            })
            
            print(f"[INFO] Outer fold {outer_fold + 1} results:")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Validation F1: {val_f1:.4f}")
            print(f"  Validation Precision: {val_precision:.4f}")
            print(f"  Validation Recall: {val_recall:.4f}")
        
        # Calculate consensus best parameters
        consensus_params = self._calculate_consensus_parameters(best_params_per_fold, synthetic_ratio, augmentation_method)
        
        # Calculate final statistics
        final_results = self._calculate_final_statistics(outer_results)
        
        # Save results
        results_summary = {
            'optimization_method': optimization_method,
            'outer_folds': outer_folds,
            'inner_folds': inner_folds,
            'n_trials': n_trials,
            'epochs': epochs,
            'sequence_length': sequence_length,
            'consensus_best_params': consensus_params,
            'outer_results': outer_results,
            'final_statistics': final_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_results(results_summary)
        
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING COMPLETED")
        print(f"{'='*80}")
        print(f"Final Statistics:")
        print(f"  Mean Accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
        print(f"  Mean F1: {final_results['mean_f1']:.4f} ± {final_results['std_f1']:.4f}")
        print(f"  Mean Precision: {final_results['mean_precision']:.4f} ± {final_results['std_precision']:.4f}")
        print(f"  Mean Recall: {final_results['mean_recall']:.4f} ± {final_results['std_recall']:.4f}")
        print(f"\nConsensus Best Parameters:")
        for param, value in consensus_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*80}")
        
        return results_summary
    
    def _tune_hyperparameters(
        self, acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
        inner_folds: int = 3, optimization_method: str = 'random_search',
        n_trials: int = 50, epochs: int = 50, sequence_length: int = 600,
        outer_fold: int = 1, synthetic_ratio: float = 100.0, 
        augmentation_method: str = 'adasyn'
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using inner cross-validation.
        """
        
        # Generate parameter combinations
        if optimization_method == 'grid_search':
            param_combinations = list(ParameterGrid(self.param_spaces['grid_search']))
        elif optimization_method == 'random_search':
            param_combinations = list(ParameterSampler(
                self.param_spaces['random_search'], 
                n_iter=n_trials, 
                random_state=42
            ))
        elif optimization_method == 'bayesian_search':
            # For now, use random search as fallback
            # TODO: Implement proper Bayesian optimization with optuna
            param_combinations = list(ParameterSampler(
                self.param_spaces['bayesian_search'], 
                n_iter=n_trials, 
                random_state=42
            ))
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        print(f"[INFO] Testing {len(param_combinations)} parameter combinations...")
        
        best_score = -np.inf
        best_params = None
        tuning_results = []
        
        for i, params in enumerate(param_combinations):
            print(f"\n[INFO] Trial {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            # Inner cross-validation
            inner_scores = self._inner_cross_validation(
                acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
                params, inner_folds, epochs, sequence_length, synthetic_ratio, augmentation_method
            )
            
            mean_score = np.mean(inner_scores)
            std_score = np.std(inner_scores)
            
            tuning_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': inner_scores
            })
            
            print(f"Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                print(f"*** New best parameters! ***")
        
        # Save tuning results for this outer fold
        tuning_file = os.path.join(self.output_dir, f'tuning_results_outer_fold_{outer_fold}.json')
        with open(tuning_file, 'w') as f:
            json.dump(tuning_results, f, indent=2, default=str)
        
        print(f"\n[INFO] Best parameters for outer fold {outer_fold}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best CV Score: {best_score:.4f}")
        
        return best_params
    
    def _inner_cross_validation(
        self, acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
        params: Dict[str, Any], inner_folds: int, epochs: int, sequence_length: int,
        synthetic_ratio: float = 100.0, augmentation_method: str = 'adasyn'
    ) -> List[float]:
        """
        Perform inner cross-validation for a single parameter combination.
        """
        
        # Combine data for inner CV
        all_data = np.concatenate([acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data])
        all_labels = np.concatenate([
            np.zeros(len(acc_can_data)),
            np.zeros(len(acc_nc_data)),
            np.ones(len(don_can_data)),
            np.ones(len(don_nc_data)),
            np.full(len(neg_data), 2)
        ])
        
        from sklearn.model_selection import KFold
        inner_kfold = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
        
        scores = []
        
        for train_idx, val_idx in inner_kfold.split(all_data):
            # Split data
            X_train, X_val = all_data[train_idx], all_data[val_idx]
            y_train, y_val = all_labels[train_idx], all_labels[val_idx]
            
            # Create and train model with current parameters
            model = create_model_with_hyperparameters(
                sequence_length=sequence_length,
                num_classes=3,
                **params
            )
            
            # Convert to categorical
            y_train_cat = to_categorical(y_train, num_classes=3)
            y_val_cat = to_categorical(y_val, num_classes=3)
            
            # Train model
            model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=params.get('batch_size', 32),
                verbose=1
            )
            
            # Evaluate
            val_predictions = model.predict(X_val, verbose=0)
            val_pred_classes = np.argmax(val_predictions, axis=1)
            
            score = accuracy_score(y_val, val_pred_classes)
            scores.append(score)
        
        return scores
    
    def _train_final_model(
        self, acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
        params: Dict[str, Any], epochs: int, sequence_length: int,
        synthetic_ratio: float = 100.0, augmentation_method: str = 'adasyn'
    ):
        """
        Train final model with given parameters.
        """
        
        # Combine all training data
        all_data = np.concatenate([acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data])
        all_labels = np.concatenate([
            np.zeros(len(acc_can_data)),
            np.zeros(len(acc_nc_data)),
            np.ones(len(don_can_data)),
            np.ones(len(don_nc_data)),
            np.full(len(neg_data), 2)
        ])
        
        # Create model
        model = create_model_with_hyperparameters(
            sequence_length=sequence_length,
            num_classes=3,
            **params
        )
        
        # Convert to categorical
        y_cat = to_categorical(all_labels, num_classes=3)
        
        # Train model
        model.fit(
            all_data, y_cat,
            epochs=epochs,
            batch_size=params.get('batch_size', 32),
            verbose=1
        )
        
        return model
    
    def _calculate_consensus_parameters(self, best_params_per_fold: List[Dict], synthetic_ratio: float = 5.0, augmentation_method: str = 'adasyn') -> Dict[str, Any]:
        """
        Calculate consensus parameters from best parameters of each fold.
        """
        
        # For numerical parameters, take the mean
        # For categorical parameters, take the most frequent value
        
        consensus = {}
        
        # Get all parameter names
        all_param_names = set()
        for params in best_params_per_fold:
            all_param_names.update(params.keys())
        
        for param_name in all_param_names:
            values = [params.get(param_name) for params in best_params_per_fold if param_name in params]
            
            if not values:
                continue
            
            # Check if parameter is numerical
            if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                # Numerical parameter - take mean
                consensus[param_name] = np.mean(values)
            else:
                # Categorical parameter - take most frequent
                from collections import Counter
                consensus[param_name] = Counter(values).most_common(1)[0][0]
        
        # Add fixed parameters that were not tuned
        consensus['synthetic_ratio'] = synthetic_ratio
        consensus['augmentation_method'] = augmentation_method
        
        return consensus
    
    def _calculate_final_statistics(self, outer_results: List[Dict]) -> Dict[str, float]:
        """
        Calculate final statistics from outer cross-validation results.
        """
        
        accuracies = [result['val_accuracy'] for result in outer_results]
        f1_scores = [result['val_f1'] for result in outer_results]
        precisions = [result['val_precision'] for result in outer_results]
        recalls = [result['val_recall'] for result in outer_results]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls)
        }
    
    def _save_results(self, results_summary: Dict[str, Any]):
        """
        Save hyperparameter tuning results.
        """
        
        # Save main results
        results_file = os.path.join(self.output_dir, 'hyperparameter_tuning_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save consensus parameters separately
        consensus_file = os.path.join(self.output_dir, 'consensus_best_parameters.json')
        with open(consensus_file, 'w') as f:
            json.dump(results_summary['consensus_best_params'], f, indent=2, default=str)
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, 'hyperparameter_tuning_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Optimization Method: {results_summary['optimization_method']}\n")
            f.write(f"Outer Folds: {results_summary['outer_folds']}\n")
            f.write(f"Inner Folds: {results_summary['inner_folds']}\n")
            f.write(f"Number of Trials: {results_summary['n_trials']}\n")
            f.write(f"Training Epochs: {results_summary['epochs']}\n")
            f.write(f"Sequence Length: {results_summary['sequence_length']}bp\n\n")
            
            f.write("FINAL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            stats = results_summary['final_statistics']
            f.write(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}\n")
            f.write(f"Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}\n")
            f.write(f"Mean Precision: {stats['mean_precision']:.4f} ± {stats['std_precision']:.4f}\n")
            f.write(f"Mean Recall: {stats['mean_recall']:.4f} ± {stats['std_recall']:.4f}\n\n")
            
            f.write("CONSENSUS BEST PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            for param, value in results_summary['consensus_best_params'].items():
                f.write(f"{param}: {value}\n")
        
        print(f"\n[INFO] Results saved to: {self.output_dir}")
        print(f"  - Main results: {results_file}")
        print(f"  - Consensus parameters: {consensus_file}")
        print(f"  - Summary report: {summary_file}")


def load_consensus_hyperparameters(results_dir: str) -> Dict[str, Any]:
    """
    Load consensus hyperparameters from hyperparameter tuning results.
    """
    
    consensus_file = os.path.join(results_dir, 'consensus_best_parameters.json')
    
    if not os.path.exists(consensus_file):
        raise FileNotFoundError(f"Consensus parameters file not found: {consensus_file}")
    
    with open(consensus_file, 'r') as f:
        consensus_params = json.load(f)
    
    return consensus_params
