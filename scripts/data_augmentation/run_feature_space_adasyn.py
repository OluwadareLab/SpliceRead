import argparse
from feature_space_adasyn import (
    extract_gc_at_features,
    adasyn_on_content_features,
    save_feature_samples
)

def generate_synthetic_feature_samples(
    canonical_acceptor_folder, noncanonical_acceptor_folder, synthetic_acceptor_folder,
    canonical_donor_folder, noncanonical_donor_folder, synthetic_donor_folder,
    n_synthetic_acceptor, n_synthetic_donor
):
    print("\n[ACCEPTOR] Generating synthetic non-canonical sequences...")
    acc_can = extract_gc_at_features(canonical_acceptor_folder, label=0)
    acc_nc = extract_gc_at_features(noncanonical_acceptor_folder, label=1)
    acc_synthetic = adasyn_on_content_features(acc_can, acc_nc, n_synthetic_acceptor)
    save_feature_samples(acc_synthetic, synthetic_acceptor_folder, prefix="synthetic_acceptor")

    print("\n[DONOR] Generating synthetic non-canonical sequences...")
    don_can = extract_gc_at_features(canonical_donor_folder, label=0)
    don_nc = extract_gc_at_features(noncanonical_donor_folder, label=1)
    don_synthetic = adasyn_on_content_features(don_can, don_nc, n_synthetic_donor)
    save_feature_samples(don_synthetic, synthetic_donor_folder, prefix="synthetic_donor")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADASYN on GC/AT content features")
    parser.add_argument('--acc_can', required=True, help='Path to canonical acceptor folder')
    parser.add_argument('--acc_nc', required=True, help='Path to non-canonical acceptor folder')
    parser.add_argument('--acc_out', required=True, help='Output folder for synthetic acceptor')

    parser.add_argument('--don_can', required=True, help='Path to canonical donor folder')
    parser.add_argument('--don_nc', required=True, help='Path to non-canonical donor folder')
    parser.add_argument('--don_out', required=True, help='Output folder for synthetic donor')

    parser.add_argument('--n_acc', type=int, required=True, help='Number of synthetic acceptor samples')
    parser.add_argument('--n_don', type=int, required=True, help='Number of synthetic donor samples')

    args = parser.parse_args()

    generate_synthetic_feature_samples(
        canonical_acceptor_folder=args.acc_can,
        noncanonical_acceptor_folder=args.acc_nc,
        synthetic_acceptor_folder=args.acc_out,
        canonical_donor_folder=args.don_can,
        noncanonical_donor_folder=args.don_nc,
        synthetic_donor_folder=args.don_out,
        n_synthetic_acceptor=args.n_acc,
        n_synthetic_donor=args.n_don
    )
