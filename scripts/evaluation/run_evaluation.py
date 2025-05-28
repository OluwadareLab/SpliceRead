import os
import argparse
from model_evaluator import load_test_data, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on canonical/non-canonical test set.")
    parser.add_argument('--model_path', required=True, help="Path to .h5 model file")
    parser.add_argument('--test_data', required=True, help="Path to test dataset root")
    parser.add_argument('--out_dir', default="evaluations", help="Directory to save evaluation metrics")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading test data from: {args.test_data}")
    X, y = load_test_data(args.test_data)

    print(f"Evaluating model: {args.model_path}")
    acc, f1, prec, rec, report = evaluate_model(args.model_path, X, y)

    out_file = os.path.join(args.out_dir, "metrics.txt")
    with open(out_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[Done] Metrics saved to: {out_file}")

if __name__ == "__main__":
    main()
