import argparse
import math
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
    parser.add_argument('--ratio', type=float, required=True, help='Desired ratio (%) of non-canonical to canonical (e.g., 100 for equal, 10 for 10%%)')
    parser.add_argument('--acc_can', required=True, help='Acceptor canonical folder')
    parser.add_argument('--acc_nc', required=True, help='Acceptor non-canonical folder')
    parser.add_argument('--don_can', required=True, help='Donor canonical folder')
    parser.add_argument('--don_nc', required=True, help='Donor non-canonical folder')
    parser.add_argument('--out_acc', required=True, help='Output folder for synthetic acceptor')
    parser.add_argument('--out_don', required=True, help='Output folder for synthetic donor')
    return parser.parse_args()

def main():
    args = parse_args()
    ratio = args.ratio

    # Load canonical and non-canonical sequences
    acc_can_seqs = load_sequences_from_folder(args.acc_can)
    acc_nc_seqs = load_sequences_from_folder(args.acc_nc)
    don_can_seqs = load_sequences_from_folder(args.don_can)
    don_nc_seqs = load_sequences_from_folder(args.don_nc)

    # Compute target counts using the formula
    acc_can_count = len(acc_can_seqs)
    acc_nc_count = len(acc_nc_seqs)
    don_can_count = len(don_can_seqs)
    don_nc_count = len(don_nc_seqs)

    acc_target = math.ceil((ratio / 100) * acc_can_count)
    don_target = math.ceil((ratio / 100) * don_can_count)

    print(f"[INFO] Acceptor: Canonical={acc_can_count}, Non-Canonical={acc_nc_count}, Target={acc_target}")
    print(f"[INFO] Donor: Canonical={don_can_count}, Non-Canonical={don_nc_count}, Target={don_target}")

    X_acc, enc_acc = sequence_to_onehot(acc_nc_seqs)
    X_don, enc_don = sequence_to_onehot(don_nc_seqs)

    for label, X, encoder, target, out_folder, real_count in [
        ("Acceptor", X_acc, enc_acc, acc_target, args.out_acc, acc_nc_count),
        ("Donor", X_don, enc_don, don_target, args.out_don, don_nc_count)
    ]:
        try:
            needed = target - real_count
            if needed <= 0:
                print(f"[INFO] {label}: Enough non-canonical samples already present, skipping generation.")
                continue
            print(f"\n[INFO] Generating {needed} synthetic {label} sequences using {'SMOTE' if args.use_smote else 'ADASYN'}...")
            if args.use_smote:
                synthetic = apply_smote(X, target)
            else:
                synthetic = apply_adasyn(X, target)

            decoded = onehot_to_sequence(synthetic[-needed:], encoder)  # Only the synthetic part
            save_sequences_to_folder(out_folder, decoded, prefix=label.lower())
        except Exception as e:
            if not args.use_smote:
                print(f"[WARNING] ADASYN failed for {label}: {e}. Retrying with SMOTE...")
                try:
                    synthetic = apply_smote(X, target)
                    decoded = onehot_to_sequence(synthetic[-needed:], encoder)
                    save_sequences_to_folder(out_folder, decoded, prefix=label.lower())
                except Exception as se:
                    print(f"[ERROR] SMOTE also failed for {label}: {se}")
            else:
                print(f"[ERROR] SMOTE failed for {label}: {e}")

if __name__ == "__main__":
    main()
