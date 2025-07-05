import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.model_evaluator import evaluate_model_three_class, load_test_data_three_class

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SpliceRead model with 3-class system')
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

    print("[INFO] Loading test data with 3-class system...")
    X_test, y_test = load_test_data_three_class(
        args.test_data, 
        show_progress=args.show_progress,
        sequence_length=args.sequence_length
    )
    
    if len(X_test) == 0:
        print("[ERROR] No test data loaded! Check the test data directory path.")
        sys.exit(1)
    
    print("[INFO] Evaluating model with 3-class system...")
    accuracy, f1, precision, recall, report = evaluate_model_three_class(args.model_path, X_test, y_test)
    
    # Save results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.out_dir, f'evaluation_results_{args.sequence_length}bp_{timestamp}.txt')
    
    with open(output_file, 'w') as f:
        f.write("SpliceRead 3-Class Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
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
        
        f.write("Results:\n")
        f.write("-" * 10 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 20 + "\n")
        f.write(report)

    # Display results
    print(f"\n" + "="*60)
    print(f"EVALUATION RESULTS ({args.sequence_length}bp)")
    print("="*60)
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nClassification Report:")
    print(report)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()