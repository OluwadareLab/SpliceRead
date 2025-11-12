
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras


try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.keras.backend.clear_session()
except Exception as e:
    print(f"[WARNING] Could not configure GPU: {e}")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


BENCHMARK_DIR = os.path.join(CURRENT_DIR, "Benchmark_splicefinder", "SpliceFinder")
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from data_utils.loader import load_test_data_three_class
from evaluation.model_evaluator import evaluate_model_three_class, ResidualBlock, load_test_data_with_canonical_info, evaluate_model_with_canonical_analysis
from cmr_ncmr_metrics import calculate_cmr_ncmr_metrics, save_cmr_ncmr_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default=None, help='Test data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--sequence_length', type=int, default=600, choices=[400, 600], help='Sequence length')
    parser.add_argument('--show_progress', action='store_true', help='Show progress bars')
    
    args = parser.parse_args()
    
    
    if args.test_dir is None:
        if args.sequence_length == 400:
            args.test_dir = './test_data_400bp'
        else:
            args.test_dir = './test_data_600bp'
    
    print(f"[INFO] Loading trained model from: {args.model_path}")
    print(f"[INFO] Loading test data from: {args.test_dir}")
    print(f"[INFO] Sequence length: {args.sequence_length}bp")
    
    
    tf.keras.backend.clear_session()
    
    
    try:
        model = keras.models.load_model(
            args.model_path, 
            custom_objects={'ResidualBlock': ResidualBlock}, 
            compile=False
        )
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"[INFO] Model loaded successfully (recompiled for inference)!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
    
    
    print(f"[INFO] Loading test data with canonical information...")
    X_test, y_test, canonical_info = load_test_data_with_canonical_info(
        args.test_dir, 
        show_progress=args.show_progress,
        sequence_length=args.sequence_length
    )
    
    print(f"[INFO] Test data loaded: {X_test.shape[0]} samples")
    print(f"[INFO] Canonical info - Acceptor_can: {len(canonical_info.get('acceptor_canonical', []))}, Acceptor_nc: {len(canonical_info.get('acceptor_noncanonical', []))}, Donor_can: {len(canonical_info.get('donor_canonical', []))}, Donor_nc: {len(canonical_info.get('donor_noncanonical', []))}")
    
    
    print(f"[INFO] Evaluating model on test data with CMR/NCMR analysis...")
    
    predictions = model.predict(X_test, verbose=1 if args.show_progress else 0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    
    accuracy = accuracy_score(y_test, predicted_classes)
    f1 = f1_score(y_test, predicted_classes, average='weighted')
    precision = precision_score(y_test, predicted_classes, average='weighted')
    recall = recall_score(y_test, predicted_classes, average='weighted')
    classification_report_str = classification_report(y_test, predicted_classes, 
                                                      target_names=['Acceptor', 'Donor', 'No Splice Site'])
    
    
    canonical_analysis = calculate_cmr_ncmr_metrics(y_test, predicted_classes, canonical_info)
    
    
    print(f"\n{'='*60}")
    print(f"TEST EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"{'='*60}")
    

    os.makedirs(args.output_dir, exist_ok=True)
    
    
    results_file = os.path.join(args.output_dir, 'test_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TEST EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_dir}\n")
        f.write(f"Sequence Length: {args.sequence_length}bp\n")
        f.write(f"Test Samples: {X_test.shape[0]}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report_str)
        
        f.write(f"\n\nCMR and NCMR Analysis:\n")
        f.write("="*60 + "\n")
        f.write("Canonical Misclassification Rate (CMR) and Non-Canonical Misclassification Rate (NCMR)\n\n")
        
        cmr_ncmr_metrics = canonical_analysis
        
        acc_can = cmr_ncmr_metrics.get("acceptor_canonical", {})
        if acc_can:
            f.write(f"Acceptor Canonical (CMR): {acc_can.get('cmr', 0.0):.4f} ({acc_can.get('misclassified', 0)}/{acc_can.get('total', 0)})\n")
            f.write(f"  Total: {acc_can.get('total', 0)}, Correct: {acc_can.get('correct', 0)}, Accuracy: {acc_can.get('accuracy', 0.0):.4f}\n\n")
        
        acc_nc = cmr_ncmr_metrics.get("acceptor_noncanonical", {})
        if acc_nc:
            f.write(f"Acceptor Non-canonical (NCMR): {acc_nc.get('ncmr', 0.0):.4f} ({acc_nc.get('misclassified', 0)}/{acc_nc.get('total', 0)})\n")
            f.write(f"  Total: {acc_nc.get('total', 0)}, Correct: {acc_nc.get('correct', 0)}, Accuracy: {acc_nc.get('accuracy', 0.0):.4f}\n\n")
        
        don_can = cmr_ncmr_metrics.get("donor_canonical", {})
        if don_can:
            f.write(f"Donor Canonical (CMR): {don_can.get('cmr', 0.0):.4f} ({don_can.get('misclassified', 0)}/{don_can.get('total', 0)})\n")
            f.write(f"  Total: {don_can.get('total', 0)}, Correct: {don_can.get('correct', 0)}, Accuracy: {don_can.get('accuracy', 0.0):.4f}\n\n")
        
        don_nc = cmr_ncmr_metrics.get("donor_noncanonical", {})
        if don_nc:
            f.write(f"Donor Non-canonical (NCMR): {don_nc.get('ncmr', 0.0):.4f} ({don_nc.get('misclassified', 0)}/{don_nc.get('total', 0)})\n")
            f.write(f"  Total: {don_nc.get('total', 0)}, Correct: {don_nc.get('correct', 0)}, Accuracy: {don_nc.get('accuracy', 0.0):.4f}\n\n")
        
        overall_cmr = 0.0
        overall_ncmr = 0.0
        
        if acc_can and don_can:
            overall_cmr = (acc_can.get('cmr', 0.0) + don_can.get('cmr', 0.0)) / 2
        
        if acc_nc and don_nc:
            overall_ncmr = (acc_nc.get('ncmr', 0.0) + don_nc.get('ncmr', 0.0)) / 2
        
        f.write(f"Overall CMR: {overall_cmr:.4f}\n")
        f.write(f"Overall NCMR: {overall_ncmr:.4f}\n")
        
        results_dict = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'cmr_ncmr_metrics': cmr_ncmr_metrics
        }
        cmr_ncmr_file = os.path.join(args.output_dir, 'cmr_ncmr_results.txt')
        save_cmr_ncmr_results(results_dict, cmr_ncmr_file)
    
    print(f"[INFO] Test evaluation results saved to: {results_file}")
    
    predictions_file = os.path.join(args.output_dir, 'test_predictions.csv')
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
            
    import pandas as pd
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred_classes,
        'confidence': np.max(y_pred, axis=1)
    })
    predictions_df.to_csv(predictions_file, index=False)
    print(f"[INFO] Predictions saved to: {predictions_file}")

if __name__ == "__main__":
    main()

