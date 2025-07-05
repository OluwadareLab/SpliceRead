import sys
import os
import argparse
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_utils.loader import load_base_data_three_class, load_base_data_three_class_separated
from training.training_utils import k_fold_cross_validation, k_fold_cross_validation_with_separated_data
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
        print(f"Synthetic Data: Generated per-fold ({args.synthetic_ratio}% ratio, {'SMOTE' if args.use_smote else 'ADASYN'})")
        print("Note: Follows run_generator.py logic - generates synthetic NC from real NC sequences")
        print("✓ Properly separates canonical/non-canonical sequences per fold")
    
    print("="*60)
    print(f"Models will be saved to: {args.model_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    print(f"Training logs (CSV) will be saved to: {args.output_dir}/training_log_fold_N.csv")
    print(f"Cross-validation folds: {args.folds}")
    print(f"Sequence length: {args.sequence_length}bp")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SpliceRead with flexible sequence length (400bp or 600bp)."
    )
    
    parser.add_argument('--three_class_no_synthetic', action='store_true',
                        help='Use 3-class system without synthetic data')
    parser.add_argument('--three_class', action='store_true',
                        help='Use 3-class system with synthetic data (same as default behavior)')
    parser.add_argument('--synthetic_ratio', type=float, default=100.0,
                        help='Ratio (%%) of non-canonical to canonical sequences for synthetic generation (default: 100.0)')
    parser.add_argument('--use_smote', action='store_true',
                        help='Use SMOTE instead of ADASYN for synthetic data generation')
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
    
    # Model and training arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model for evaluation (if not provided, uses model_dir/best_model_fold_N.h5)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    
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
    
    if args.model_dir is None:
        if args.sequence_length == 400:
            args.model_dir = f'./model_output_{args.sequence_length}bp'
        else:
            args.model_dir = f'./model_output_{args.sequence_length}bp'
    
    if args.output_dir is None:
        if args.sequence_length == 400:
            args.output_dir = f'./results_{args.sequence_length}bp'
        else:
            args.output_dir = f'./results_{args.sequence_length}bp'
    
    display_data_configuration(args)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.output_dir}")

    # ========== TRAINING ==========
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
            use_smote=False,
            X_synthetic=None,
            y_synthetic=None,
            output_dir=args.output_dir
        )
        
    else:
        print(f"[INFO] Loading data with canonical/non-canonical separation for proper synthetic generation...")
        print(f"[INFO] Will generate synthetic data per fold ({args.synthetic_ratio}% ratio) following run_generator.py logic")
        
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
        print(f"[INFO] Starting {args.folds}-fold cross-validation training (WITH synthetic)...")
        
        k_fold_cross_validation_with_separated_data(
            acc_can_data, acc_nc_data, don_can_data, don_nc_data, neg_data,
            k=args.folds,
            model_dir=args.model_dir,
            use_synthetic=True,
            synthetic_ratio=args.synthetic_ratio,
            use_smote=args.use_smote,
            output_dir=args.output_dir
        )

    # ========== EVALUATION ==========
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
    
    # Display results
    mode_name = "NO SYNTHETIC" if args.three_class_no_synthetic else "WITH SYNTHETIC"
    print(f"\n========== 3-CLASS EVALUATION RESULTS ({mode_name}) - {args.sequence_length}bp ==========")
    print("Classes: 0=Acceptor (all ACC), 1=Donor (all DON), 2=No Splice Site (all NEG)")
    if not args.three_class_no_synthetic:
        print(f"Synthetic Data: Generated per fold at {args.synthetic_ratio}% ratio using {'SMOTE' if args.use_smote else 'ADASYN'}")
    
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}\n")
    print("Classification Report:\n")
    print(report)

    # Save evaluation results to files
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    system_type = "3class_nosynthetic" if args.three_class_no_synthetic else f"3class_synthetic_{args.synthetic_ratio}pct"
    
    # Save summary results
    results_file = os.path.join(args.output_dir, f"evaluation_results_{system_type}_{args.sequence_length}bp_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"SPLICEREAD EVALUATION RESULTS ({args.sequence_length}bp SEQUENCES)\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data Directory: {args.test_dir}\n")
        f.write(f"Test Data Shape: {X_test.shape}\n")
        f.write(f"Labels Shape: {y_test.shape}\n")
        f.write(f"Sequence Length: {args.sequence_length}bp\n\n")
        
        f.write("SYSTEM: 3-Class\n")
        f.write("Classes: 0=Acceptor (all ACC), 1=Donor (all DON), 2=No Splice Site (all NEG)\n")
        if not args.three_class_no_synthetic:
            f.write(f"Synthetic Data: Generated per fold at {args.synthetic_ratio}% ratio using {'SMOTE' if args.use_smote else 'ADASYN'}\n")
        else:
            f.write("Synthetic Data: EXCLUDED\n")
        
        f.write(f"\nRESULTS:\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"[INFO] Results saved to: {results_file}")

if __name__ == "__main__":
    main() 