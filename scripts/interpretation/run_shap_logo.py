import argparse
from shap_logo_generator import run_shap_weighted_logomaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Weighted Sequence Logo Generator")
    parser.add_argument('--model', required=True, help="Path to trained Keras model (.h5)")
    parser.add_argument('--data', required=True, help="Path to dataset folder")
    parser.add_argument('--samples', type=int, default=100, help="Number of sequences to analyze")
    parser.add_argument('--class_index', type=int, default=1, help="Class index to explain")
    parser.add_argument('--output', default="shap_weighted_logo.png", help="Output image filename")
    args = parser.parse_args()

    run_shap_weighted_logomaker(args.model, args.data, args.samples, args.class_index, args.output)
