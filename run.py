import sys
import os
import argparse

# Setup Python path to include scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "scripts"))

from data_utils.loader import load_data_from_folder
from training.training_utils import k_fold_cross_validation
from evaluation.model_evaluator import evaluate_model

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run training and evaluation for SpliceRead")
    parser.add_argument('--include_neg', action='store_true', help='Include NEG (negative) sequences in training/testing')
    parser.add_argument('--include_synthetic', action='store_true', help='Include synthetic (SYN) sequences in training/testing')
    args = parser.parse_args()

    # Load training data
    print(f"[INFO] Loading training data (include_neg={args.include_neg}, include_synthetic={args.include_synthetic})...")
    X, y = load_data_from_folder('data/train', include_neg=args.include_neg, include_synthetic=args.include_synthetic)
    k_fold_cross_validation(X, y, k=5)

    # Load test data
    print(f"\n[INFO] Loading test data (include_neg={args.include_neg}, include_synthetic={args.include_synthetic})...")
    X_test, y_test = load_data_from_folder('data/test', include_neg=args.include_neg, include_synthetic=args.include_synthetic)
    acc, f1, precision, recall, report = evaluate_model('model_files/best_model_fold_5.h5', X_test, y_test)


    # Print metrics
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n")
    print(report)
