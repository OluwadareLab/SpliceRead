import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.model_evaluator import evaluate_model_with_canonical_analysis, load_test_data_with_canonical_info

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SpliceRead model with canonical/non-canonical analysis')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (.h5 file)')
    parser.add_argument('--test_data', type=str, required=True, 
                       help='Path to test data directory')
    parser.add_argument('--out_dir', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--sequence_length', type=int, default=600, choices=[400, 600],
                       help='Sequence length in base pairs (default: 600)')
    parser.add_argument('--show_progress', action='store_true',
                       help='Show progress bars while loading data')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Evaluating model: {args.model_path}")
    print(f"[INFO] Test data directory: {args.test_data}")
    print(f"[INFO] Sequence length: {args.sequence_length}bp")
    print(f"[INFO] Output directory: {args.out_dir}")
    print("-" * 60)

    print("[INFO] Loading test data with canonical/non-canonical information...")
    X_test, y_test, canonical_info = load_test_data_with_canonical_info(
        args.test_data, 
        show_progress=args.show_progress,
        sequence_length=args.sequence_length
    )
    
    if len(X_test) == 0:
        print("[ERROR] No test data loaded! Check the test data directory path.")
        sys.exit(1)
    
    print("[INFO] Evaluating model with canonical/non-canonical analysis...")
    accuracy, f1, precision, recall, report, canonical_analysis = evaluate_model_with_canonical_analysis(
        args.model_path, X_test, y_test, canonical_info
    )
    
    # Save results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.out_dir, f'evaluation_canonical_{args.sequence_length}bp_{timestamp}.txt')
    
    with open(output_file, 'w') as f:
        f.write("SpliceRead Evaluation with Canonical/Non-canonical Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Sequence Length: {args.sequence_length}bp\n")
        f.write(f"Test Data Shape: {X_test.shape}\n")
        f.write(f"Labels Shape: {y_test.shape}\n\n")
        
        f.write("3-Class System:\n")
        f.write("  0: Acceptor (ACC/CAN + ACC/NC)\n")
        f.write("  1: Donor (DON/CAN + DON/NC)\n")
        f.write("  2: No Splice Site (NEG/ACC + NEG/DON)\n\n")
        
        f.write("Overall Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 25 + "\n")
        f.write(report + "\n\n")
        
        f.write("Canonical/Non-canonical Analysis:\n")
        f.write("=" * 40 + "\n\n")
        
        # Acceptor Analysis
        if 'acceptor_canonical' in canonical_analysis:
            acc_can = canonical_analysis['acceptor_canonical']
            f.write("Acceptor Canonical:\n")
            f.write(f"  Total sequences: {acc_can['total']}\n")
            f.write(f"  Correct predictions: {acc_can['correct']}\n")
            f.write(f"  Misclassified: {acc_can['misclassified']}\n")
            f.write(f"  Accuracy: {acc_can['accuracy']:.4f}\n")
            f.write(f"  Misclassification rate: {acc_can['misclassification_rate']:.4f}\n\n")
        
        if 'acceptor_noncanonical' in canonical_analysis:
            acc_nc = canonical_analysis['acceptor_noncanonical']
            f.write("Acceptor Non-canonical:\n")
            f.write(f"  Total sequences: {acc_nc['total']}\n")
            f.write(f"  Correct predictions: {acc_nc['correct']}\n")
            f.write(f"  Misclassified: {acc_nc['misclassified']}\n")
            f.write(f"  Accuracy: {acc_nc['accuracy']:.4f}\n")
            f.write(f"  Misclassification rate: {acc_nc['misclassification_rate']:.4f}\n\n")
        
        # Donor Analysis
        if 'donor_canonical' in canonical_analysis:
            don_can = canonical_analysis['donor_canonical']
            f.write("Donor Canonical:\n")
            f.write(f"  Total sequences: {don_can['total']}\n")
            f.write(f"  Correct predictions: {don_can['correct']}\n")
            f.write(f"  Misclassified: {don_can['misclassified']}\n")
            f.write(f"  Accuracy: {don_can['accuracy']:.4f}\n")
            f.write(f"  Misclassification rate: {don_can['misclassification_rate']:.4f}\n\n")
        
        if 'donor_noncanonical' in canonical_analysis:
            don_nc = canonical_analysis['donor_noncanonical']
            f.write("Donor Non-canonical:\n")
            f.write(f"  Total sequences: {don_nc['total']}\n")
            f.write(f"  Correct predictions: {don_nc['correct']}\n")
            f.write(f"  Misclassified: {don_nc['misclassified']}\n")
            f.write(f"  Accuracy: {don_nc['accuracy']:.4f}\n")
            f.write(f"  Misclassification rate: {don_nc['misclassification_rate']:.4f}\n\n")

    # Display results
    print(f"\n" + "="*60)
    print(f"EVALUATION RESULTS WITH CANONICAL ANALYSIS ({args.sequence_length}bp)")
    print("="*60)
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("\nCanonical/Non-canonical Analysis:")
    print("=" * 40)
    
    # Display canonical analysis results
    if 'acceptor_canonical' in canonical_analysis:
        acc_can = canonical_analysis['acceptor_canonical']
        print(f"Acceptor Canonical:")
        print(f"  Total: {acc_can['total']}, Correct: {acc_can['correct']}, Misclassified: {acc_can['misclassified']}")
        print(f"  Accuracy: {acc_can['accuracy']:.4f}, Misclassification rate: {acc_can['misclassification_rate']:.4f}")
    
    if 'acceptor_noncanonical' in canonical_analysis:
        acc_nc = canonical_analysis['acceptor_noncanonical']
        print(f"Acceptor Non-canonical:")
        print(f"  Total: {acc_nc['total']}, Correct: {acc_nc['correct']}, Misclassified: {acc_nc['misclassified']}")
        print(f"  Accuracy: {acc_nc['accuracy']:.4f}, Misclassification rate: {acc_nc['misclassification_rate']:.4f}")
    
    if 'donor_canonical' in canonical_analysis:
        don_can = canonical_analysis['donor_canonical']
        print(f"Donor Canonical:")
        print(f"  Total: {don_can['total']}, Correct: {don_can['correct']}, Misclassified: {don_can['misclassified']}")
        print(f"  Accuracy: {don_can['accuracy']:.4f}, Misclassification rate: {don_can['misclassification_rate']:.4f}")
    
    if 'donor_noncanonical' in canonical_analysis:
        don_nc = canonical_analysis['donor_noncanonical']
        print(f"Donor Non-canonical:")
        print(f"  Total: {don_nc['total']}, Correct: {don_nc['correct']}, Misclassified: {don_nc['misclassified']}")
        print(f"  Accuracy: {don_nc['accuracy']:.4f}, Misclassification rate: {don_nc['misclassification_rate']:.4f}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()