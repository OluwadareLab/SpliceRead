import sys
import os
import argparse
import random
import numpy as np
from datetime import datetime
import tensorflow as tf


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


try:
    tf.config.experimental.enable_op_determinism()
    print(f"[INFO] Enabled TensorFlow deterministic operations for reproducibility (seed={SEED})")
except Exception as e:
    print(f"[WARNING] Could not enable deterministic operations: {e}")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_utils.loader import load_base_data_three_class, load_base_data_three_class_separated
from training.training_utils import k_fold_cross_validation, k_fold_cross_validation_with_separated_data, find_best_fold_from_cv_results, load_consensus_hyperparameters, train_final_model_on_full_dataset, train_final_model_on_full_dataset_fixed
from training.hyperparameter_tuning import HyperparameterTuner, load_consensus_hyperparameters as load_tuning_consensus
from evaluation.model_evaluator import evaluate_model_three_class
from data_utils.loader import load_test_data_three_class

def display_data_configuration(args):
    """Display what data will be loaded based on the arguments."""
    print("\n" + "="*60)
    print(f"DATA CONFIGURATION SUMMARY ({args.sequence_length}bp SEQUENCES)")
    print("="*60)
    
    if args.three_class_no_synthetic:
        print("Training Mode: 3-class (NO synthetic data)")
        print("Training/Evaluation Classes:")
        print("  0: Acceptor (ACC/CAN + ACC/NC only)")
        print("  1: Donor (DON/CAN + DON/NC only)")
        print("  2: No Splice Site (NEG/ACC + NEG/DON)")
        print("Synthetic Data: EXCLUDED")
    else:  
        print("Training Mode: 3-class (with synthetic data)")
        print("Training/Evaluation Classes:")
        print("  0: Acceptor (ACC/CAN + ACC/NC)")
        print("  1: Donor (DON/CAN + DON/NC)")
        print("  2: No Splice Site (NEG/ACC + NEG/DON)")
        
    
    print("="*60)
    print(f"Models will be saved to: {args.model_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    print(f"Training logs (CSV) will be saved to: {args.output_dir}/training_log_fold_N.csv")
    print(f"Cross-validation folds: {args.folds}")
    print(f"Training epochs: {args.epochs}")
    print(f"Sequence length: {args.sequence_length}bp")
    print("="*60 + "\n")

def _coerce_final_hparams(hp: dict) -> dict:
    """Coerce tuned hyperparameters to proper dtypes for final training.
    Ensures integers where the model/training loop expects ints and floats where needed.
    """
    if hp is None:
        return hp
    float_whitelist = {"learning_rate", "l2_reg", "dropout_rate", "synthetic_ratio"}
    
    explicit_ints = [
        "num_filters", "num_conv_layers", "kernel_size", "batch_size",
        "early_stopping_patience", "steps_per_epoch", "patience"
    ]
    for k in explicit_ints:
        if k in hp and hp[k] is not None:
            try:
                hp[k] = int(round(float(hp[k])))
            except Exception:
                pass
    for k in float_whitelist:
        if k in hp and hp[k] is not None:
            try:
                hp[k] = float(hp[k])
            except Exception:
                pass
   
    for k, v in list(hp.items()):
        if k in float_whitelist:
            continue
        try:
            fv = float(v)
            
            if abs(fv - round(fv)) < 1e-6 or any(s in k for s in ["num", "count", "size", "layers", "filters", "batch", "epochs", "patience"]):
                hp[k] = int(round(fv))
        except Exception:
            pass
    return hp

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SpliceRead with flexible sequence length (400bp or 600bp)."
    )
    
    parser.add_argument('--three_class_no_synthetic', action='store_true',
                        help='Use 3-class system without synthetic data')
    parser.add_argument('--three_class', action='store_true',
                        help='Use 3-class system with synthetic data (same as default behavior)')
    parser.add_argument('--synthetic_ratio', type=float, default=5.0,
                        help='Ratio (%%) of non-canonical to canonical sequences for synthetic generation (default: 5.0)')
    parser.add_argument('--augmentation_method', type=str, default='adasyn',
                        choices=['adasyn', 'svm_adasyn'],
                        help='Data augmentation method to use (default: adasyn)')
    parser.add_argument('--show_progress', action='store_true',
                        help='Show progress bars while loading data')
    
    # Sequence length argument
    parser.add_argument('--sequence_length', type=int, default=600, choices=[400, 600],
                        help='Sequence length in base pairs (default: 600, choices: 400, 600)')
    
    # Directory arguments - Flexible based on sequence length
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Directory with training data (auto-detected based on sequence_length if not specified)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory with test data (auto-detected based on sequence_length if not specified)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory to save/load models (auto-generated based on sequence_length if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results and outputs (auto-generated based on sequence_length if not specified)')
    
    parser.add_argument("--evaluate_test", action="store_true",help="Evaluate on test data after training (use with caution to avoid data leakage)")
    # Training mode arguments
    parser.add_argument("--mode", type=str, default="cv", choices=["cv", "final", "hyperparameter_tuning"],
                        help="Training mode: cv (5-fold cross-validation), final (train on full dataset), or hyperparameter_tuning (nested CV with hyperparameter optimization)")
    parser.add_argument("--cv_results_dir", type=str, default=None,
                        help="Directory containing cross-validation results (required for final mode)")
    parser.add_argument("--best_fold", type=int, default=None,
                        help="Best fold number to use for hyperparameters (auto-detected if not specified)")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model for evaluation (if not provided, uses model_dir/best_model_fold_N.h5)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=70,
                        help='Number of training epochs (default: 70)')
    
    # Hyperparameter tuning arguments
    parser.add_argument('--outer_folds', type=int, default=5,
                        help='Number of outer CV folds for hyperparameter tuning (default: 5)')
    parser.add_argument('--inner_folds', type=int, default=3,
                        help='Number of inner CV folds for hyperparameter tuning (default: 3)')
    parser.add_argument('--optimization_method', type=str, default='random_search',
                        choices=['grid_search', 'random_search', 'bayesian_search'],
                        help='Hyperparameter optimization method (default: random_search)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials for random/bayesian search (default: 50)')
    parser.add_argument('--tuning_epochs', type=int, default=50,
                        help='Number of epochs for hyperparameter tuning (default: 50)')
    parser.add_argument('--tuning_results_dir', type=str, default=None,
                        help='Directory containing hyperparameter tuning results (auto-generated if not specified)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd', 'adamax'],
                        help='Override optimizer from tuning results (default: adam, choices: adam, rmsprop, sgd, adamax)')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                        help='Override number of convolutional layers from tuning results (default: 3)')
    parser.add_argument('--kernel_size', type=int, default=9,
                        help='Override kernel size from tuning results (default: 9)')
    parser.add_argument('--num_filters', type=int, default=50,
                        help='Override number of filters from tuning results (default: 50)')
    
    # Weight parameters for sample weighting
    parser.add_argument('--acc_nc_weight', type=float, default=20.0,
                        help='Sample weight multiplier for acceptor non-canonical samples (default: 20.0)')
    parser.add_argument('--don_nc_weight', type=float, default=5.0,
                        help='Sample weight multiplier for donor non-canonical samples (default: 5.0)')
    parser.add_argument('--acc_target_nc_ratio', type=float, default=0.3,
                        help='Target oversampling ratio for acceptor non-canonical vs canonical (default: 0.3)')
    parser.add_argument('--don_target_nc_ratio', type=float, default=0.1,
                        help='Target oversampling ratio for donor non-canonical vs canonical (default: 0.1)')
    
    # Focal loss parameters (for future use - currently not implemented)
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                        help='Focal loss gamma parameter (default: 1.5, currently not used)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter (default: 0.25, currently not used)')
    
    args = parser.parse_args()
    
    
    if args.three_class and args.three_class_no_synthetic:
        print("[ERROR] Cannot use both --three_class and --three_class_no_synthetic")
        sys.exit(1)
    
    
    if args.train_dir is None:
        if args.sequence_length == 400:
            args.train_dir = './to_zenodo/data_400bp/train'
        else:
            args.train_dir = './to_zenodo/data_600bp/train'
    
    if args.test_dir is None:
        if args.sequence_length == 400:
            args.test_dir = './to_zenodo/data_400bp/test'
        else:
            args.test_dir = './to_zenodo/data_600bp/test'
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.model_dir is None:
        args.model_dir = f'./model_output_{args.sequence_length}bp_{timestamp}'
    else:
       
        args.model_dir = f'{args.model_dir}_{timestamp}'
    
    if args.output_dir is None:
        args.output_dir = f'./results_{args.sequence_length}bp_{timestamp}'
    else:
        
        args.output_dir = f'{args.output_dir}_{timestamp}'
    
    
    if args.tuning_results_dir is None:
        args.tuning_results_dir = f'./hyperparameter_tuning_results_{args.sequence_length}bp'
    
    display_data_configuration(args)

    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"[INFO] Model directory: {args.model_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")

    
    print(f"[INFO] Loading training data with 3-class system ({args.sequence_length}bp sequences)...")
    
    if args.three_class_no_synthetic:
        
        print(f"[INFO] Loading base data without synthetic sequences...")
        X_base, y_base = load_base_data_three_class(
            args.train_dir,
            show_progress=args.show_progress,
            sequence_length=args.sequence_length
        )
        
        print(f"[INFO] Training data shape: {X_base.shape}, Labels shape: {y_base.shape}")
        print(f"[INFO] Starting {args.folds}-fold cross-validation training (NO synthetic)...")
        
        k_fold_cross_validation(
            X_base, y_base, 
            k=args.folds, 
            model_dir=args.model_dir,
            use_synthetic=False,
            synthetic_ratio=0.0,
            augmentation_method='adasyn',  #
            X_synthetic=None,
            y_synthetic=None,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        
    else:
        
        print(f"[INFO] Loading data with canonical/non-canonical separation for proper synthetic generation...")
        print(f"[INFO] Will generate synthetic data per fold ({args.synthetic_ratio}% ratio) using {args.augmentation_method.upper()}")
        
        # Load separated data for proper per-fold generation
        (acc_can_data, acc_can_labels), (acc_nc_data, acc_nc_labels), \
        (don_can_data, don_can_labels), (don_nc_data, don_nc_labels), \
        (neg_data, neg_labels) = load_base_data_three_class_separated(
            args.train_dir,
            show_progress=args.show_progress,
            sequence_length=args.sequence_length
        )
        
      
        all_base_data = np.concatenate([acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data])
        all_base_labels = np.concatenate([acc_can_labels, acc_nc_labels, don_can_labels, don_nc_labels, neg_labels])
        
        print(f"[INFO] Training data shape: {all_base_data.shape}, Labels shape: {all_base_labels.shape}")
    
    if args.mode == "hyperparameter_tuning":
       e
        print(f"\n[INFO] HYPERPARAMETER TUNING MODE")
        print(f"[INFO] This will perform nested cross-validation with hyperparameter optimization")
        print(f"[INFO] Warning: This may take a very long time!")
        
        
        if args.three_class_no_synthetic:
            print(f"[ERROR] Hyperparameter tuning requires synthetic data generation")
            print(f"[ERROR] Please use --three_class (default) instead of --three_class_no_synthetic")
            sys.exit(1)
        
        print(f"[INFO] Loading data with canonical/non-canonical separation...")
        (acc_can_data, acc_can_labels), (acc_nc_data, acc_nc_labels), \
        (don_can_data, don_can_labels), (don_nc_data, don_nc_labels), \
        (neg_data, neg_labels) = load_base_data_three_class_separated(
            args.train_dir,
            show_progress=args.show_progress,
            sequence_length=args.sequence_length
        )
        
        print(f"[INFO] Starting hyperparameter tuning...")
        print(f"[INFO] Outer folds: {args.outer_folds}")
        print(f"[INFO] Inner folds: {args.inner_folds}")
        print(f"[INFO] Optimization method: {args.optimization_method}")
        print(f"[INFO] Number of trials: {args.n_trials}")
        print(f"[INFO] Tuning epochs: {args.tuning_epochs}")
        
       
        tuner = HyperparameterTuner(output_dir=args.tuning_results_dir)
        
        
        tuning_results = tuner.nested_cross_validation_with_hyperparameter_tuning(
            acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            optimization_method=args.optimization_method,
            n_trials=args.n_trials,
            epochs=args.tuning_epochs,
            sequence_length=args.sequence_length,
            synthetic_ratio=args.synthetic_ratio,
            augmentation_method=args.augmentation_method
        )
        
        print(f"\n[INFO] Hyperparameter tuning completed!")
        print(f"[INFO] Results saved to: {args.tuning_results_dir}")
        print(f"[INFO] Use --mode final --tuning_results_dir {args.tuning_results_dir} to train final model")
        
        return  
    
    elif args.mode == "final":
        
        print(f"\n[INFO] FINAL MODEL TRAINING MODE")
        print(f"[INFO] Looking for cross-validation results in: {args.cv_results_dir or args.output_dir}")
        
        
       
        cv_results_dir = args.cv_results_dir if args.cv_results_dir else args.output_dir
        
        
        if args.tuning_results_dir and os.path.exists(args.tuning_results_dir):
            print(f"[INFO] Using hyperparameter tuning results from: {args.tuning_results_dir}")
            try:
                hyperparameters = load_tuning_consensus(args.tuning_results_dir)
                hyperparameters = _coerce_final_hparams(hyperparameters)
                print(f"[INFO] Loaded consensus hyperparameters from hyperparameter tuning (coerced dtypes)")
               
                hyperparameters['synthetic_ratio'] = args.synthetic_ratio
                hyperparameters['augmentation_method'] = args.augmentation_method
                print(f"[INFO] Using synthetic_ratio={args.synthetic_ratio}% and augmentation_method={args.augmentation_method} from command line")
            except FileNotFoundError:
                print(f"[WARNING] Hyperparameter tuning results not found, falling back to CV results")
                hyperparameters = load_consensus_hyperparameters(cv_results_dir, args.synthetic_ratio, args.augmentation_method, args.epochs)
                hyperparameters = _coerce_final_hparams(hyperparameters)
        else:
            
            best_fold = None
            if args.best_fold is None:
                try:
                    best_fold = find_best_fold_from_cv_results(cv_results_dir)
                except FileNotFoundError:
                    print(f"[WARNING] No CV results found in {cv_results_dir}")
                    print(f"[INFO] Will use consensus hyperparameters (defaults)")
                    best_fold = None
            else:
                best_fold = args.best_fold
                print(f"[INFO] Using specified best fold: {best_fold}")
            
            
            hyperparameters = load_consensus_hyperparameters(cv_results_dir, args.synthetic_ratio, args.augmentation_method, args.epochs)
            hyperparameters = _coerce_final_hparams(hyperparameters)
        
       
        print(f"[INFO] Applying fixed hyperparameters for final model training:")
        
        
        if hasattr(args, 'optimizer') and args.optimizer:
            old_val = hyperparameters.get('optimizer', 'default')
            hyperparameters['optimizer'] = args.optimizer
            if old_val != args.optimizer:
                print(f"[INFO] Setting optimizer to {args.optimizer} (was: {old_val})")
        
       
        if hasattr(args, 'num_conv_layers') and args.num_conv_layers is not None:
            old_val = hyperparameters.get('num_conv_layers', 'default')
            hyperparameters['num_conv_layers'] = args.num_conv_layers
            if old_val != args.num_conv_layers:
                print(f"[INFO] Setting num_conv_layers to {args.num_conv_layers} (was: {old_val})")
        
       
        if hasattr(args, 'kernel_size'):
            old_val = hyperparameters.get('kernel_size', 'default')
            hyperparameters['kernel_size'] = args.kernel_size
            if old_val != args.kernel_size:
                print(f"[INFO] Setting kernel_size to {args.kernel_size} (was: {old_val})")
        
       
        if hasattr(args, 'num_filters'):
            old_val = hyperparameters.get('num_filters', 'default')
            hyperparameters['num_filters'] = args.num_filters
            if old_val != args.num_filters:
                print(f"[INFO] Setting num_filters to {args.num_filters} (was: {old_val})")
        
        
        hyperparameters['dropout_rate'] = 0.3
        hyperparameters['learning_rate'] = 0.0001
        hyperparameters['dense_units'] = 100
        hyperparameters['l2_reg'] = 0.0
        hyperparameters['use_batch_norm'] = False
        print(f"[INFO] Fixed values: dropout_rate=0.3, learning_rate=0.0001, dense_units=100, l2_reg=0.0, use_batch_norm=False")
        
        
        final_model, history, final_results = train_final_model_on_full_dataset_fixed(
            acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
            hyperparameters, args.model_dir, args.output_dir, args.epochs, args.sequence_length,
            acc_nc_weight=args.acc_nc_weight,
            don_nc_weight=args.don_nc_weight,
            acc_target_nc_ratio=args.acc_target_nc_ratio,
            don_target_nc_ratio=args.don_target_nc_ratio
        )
        
        print(f"\n[INFO] Final model training completed!")
        print(f"[INFO] Final model saved to: {os.path.join(args.model_dir, 'final_model.h5')}")
        print(f"[INFO] Training results saved to: {args.output_dir}")
        
       
        print(f"\n[INFO] Running automatic test evaluation...")
        
        final_model_path = os.path.join(args.model_dir, "final_model.h5")
        
        if not os.path.isfile(final_model_path):
            print(f"[ERROR] Final model file not found at: {final_model_path}")
            print(f"[INFO] Make sure training completed and the model was saved to {args.model_dir}")
            sys.exit(1)
        
       
        if args.test_dir is None:
            if args.sequence_length == 400:
                args.test_dir = './to_zenodo/data_400bp/test'
            else:
                args.test_dir = './to_zenodo/data_600bp/test'
        
        
        import subprocess
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.py")
        
        eval_cmd = [
            sys.executable, eval_script,
            '--model_path', final_model_path,
            '--test_dir', args.test_dir,
            '--output_dir', args.output_dir,
            '--sequence_length', str(args.sequence_length)
        ]
        
        if args.show_progress:
            eval_cmd.append('--show_progress')
        
        print(f"[INFO] Running evaluation: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, check=True)
        print(f"[INFO] Test evaluation completed successfully!")
        
        return  
    
    
    print(f"\n[INFO] CROSS-VALIDATION MODE")

    print(f"[INFO] Starting {args.folds}-fold cross-validation training (WITH synthetic)...")
    
    k_fold_cross_validation_with_separated_data(
        acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
        k=args.folds,
        model_dir=args.model_dir,
        use_synthetic=True,
        synthetic_ratio=args.synthetic_ratio,
        augmentation_method=args.augmentation_method,
        output_dir=args.output_dir,
            epochs=args.epochs
        )

    # ========== TEST EVALUATION (OPTIONAL FOR CV MODE) ==========
    if args.evaluate_test:
        
        print(f"\n[WARNING] Test evaluation enabled. This should only be used for final reporting!")
        print(f"[WARNING] Make sure you have not used test data for model selection or hyperparameter tuning!")
        
        
        if args.model_path is None:
            args.model_path = os.path.join(args.model_dir, f"best_model_fold_{args.folds}.h5")
        
        if not os.path.isfile(args.model_path):
            print(f"[ERROR] Model file not found at: {args.model_path}")
            print(f"[INFO] Make sure training completed and the model was saved to {args.model_dir}")
            sys.exit(1)
        
        print(f"[INFO] Evaluating model: {args.model_path}")
        print(f"[INFO] Loading test data with 3-class system ({args.sequence_length}bp sequences)...")
        
        X_test, y_test = load_test_data_three_class(
            args.test_dir, 
            show_progress=args.show_progress,
            sequence_length=args.sequence_length
        )
        
        print("[INFO] Evaluating with 3-class system...")
        acc, f1, precision, recall, report = evaluate_model_three_class(args.model_path, X_test, y_test)
        
        
        mode_name = "NO SYNTHETIC" if args.three_class_no_synthetic else "WITH SYNTHETIC"
        print(f"\n========== 3-CLASS EVALUATION RESULTS ({mode_name}) - {args.sequence_length}bp ==========")
        print("Classes: 0=Acceptor (all ACC), 1=Donor (all DON), 2=No Splice Site (all NEG)")
        if not args.three_class_no_synthetic:
            print(f"Synthetic Data: Generated per fold at {args.synthetic_ratio}% ratio using {args.augmentation_method.upper()}")
        print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"\nClassification Report:\n{report}")
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synthetic_info = f"synthetic_{args.synthetic_ratio}pct_{args.augmentation_method}" if not args.three_class_no_synthetic else "no_synthetic"
        results_file = os.path.join(args.output_dir, f"evaluation_results_3class_{synthetic_info}_{args.sequence_length}bp_{timestamp}.txt")
        
        with open(results_file, 'w') as f:
            f.write(f"3-CLASS EVALUATION RESULTS ({mode_name}) - {args.sequence_length}bp\n")
            f.write("="*60 + "\n")
            f.write(f"Classes: 0=Acceptor (all ACC), 1=Donor (all DON), 2=No Splice Site (all NEG)\n")
            if not args.three_class_no_synthetic:
                f.write(f"Synthetic Data: Generated per fold at {args.synthetic_ratio}% ratio using {args.augmentation_method.upper()}\n")
            f.write(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}\n")
            f.write(f"Accuracy : {acc:.4f}\n")
            f.write(f"F1 Score : {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall   : {recall:.4f}\n")
            f.write(f"\nClassification Report:\n{report}\n")
        
        print(f"[INFO] Results saved to: {results_file}")
    else:
        if args.mode != 'final':
            print(f"\n[INFO] Test evaluation skipped (use --evaluate_test to enable)")
            print(f"[INFO] This prevents data leakage. Use test data only for final reporting.")

if __name__ == "__main__":
    main()
