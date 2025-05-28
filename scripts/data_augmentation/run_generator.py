import argparse
from generator import (
    load_sequences_from_folder,
    sequence_to_onehot,
    onehot_to_sequence,
    apply_adasyn,
    apply_smote,
    save_sequences_to_folder
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic splice site sequences using ADASYN or SMOTE.")
    parser.add_argument('--use-smote', action='store_true', help='Use SMOTE instead of ADASYN')
    return parser.parse_args()

def main():
    args = parse_args()

    ACCEPTOR_FOLDER = 'test_input/POS/ACC/NC'
    DONOR_FOLDER = 'test_input/POS/DON/NC'
    ACCEPTOR_TARGET_COUNT = 20
    DONOR_TARGET_COUNT = 20
    OUTPUT_ACCEPTOR_FOLDER = 'test_output/POS/ACC/ADASYN_TEST'
    OUTPUT_DONOR_FOLDER = 'test_output/POS/DON/ADASYN_TEST'

    acceptor_sequences = load_sequences_from_folder(ACCEPTOR_FOLDER)
    donor_sequences = load_sequences_from_folder(DONOR_FOLDER)

    X_acceptor, encoder_acceptor = sequence_to_onehot(acceptor_sequences)
    X_donor, encoder_donor = sequence_to_onehot(donor_sequences)

    for label, X, encoder, target, out_folder in [
        ("Acceptor", X_acceptor, encoder_acceptor, ACCEPTOR_TARGET_COUNT, OUTPUT_ACCEPTOR_FOLDER),
        ("Donor", X_donor, encoder_donor, DONOR_TARGET_COUNT, OUTPUT_DONOR_FOLDER)
    ]:
        try:
            print(f"\n[INFO] Generating synthetic {label} sequences using {'SMOTE' if args.use_smote else 'ADASYN'}...")
            if args.use_smote:
                synthetic = apply_smote(X, target)
            else:
                synthetic = apply_adasyn(X, target)

            decoded = onehot_to_sequence(synthetic, encoder)
            save_sequences_to_folder(out_folder, decoded, prefix=label.lower())
        except Exception as e:
            if not args.use_smote:
                print(f"[WARNING] ADASYN failed for {label}: {e}. Retrying with SMOTE...")
                try:
                    synthetic = apply_smote(X, target)
                    decoded = onehot_to_sequence(synthetic, encoder)
                    save_sequences_to_folder(out_folder, decoded, prefix=label.lower())
                except Exception as se:
                    print(f"[ERROR] SMOTE also failed for {label}: {se}")
            else:
                print(f"[ERROR] SMOTE failed for {label}: {e}")

if __name__ == "__main__":
    main()
