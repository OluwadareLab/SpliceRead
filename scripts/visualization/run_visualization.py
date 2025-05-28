import argparse
from nucleotide_content_plotter import create_plots

def main():
    parser = argparse.ArgumentParser(description="Plot nucleotide content of canonical, non-canonical, and synthetic sequences.")
    parser.add_argument('--canonical', required=True, help="Path to canonical sequence folder")
    parser.add_argument('--noncanonical', required=True, help="Path to non-canonical sequence folder")
    parser.add_argument('--synthetic', required=True, help="Path to synthetic sequence folder")
    parser.add_argument('--title', required=True, help="Title prefix for plot files (e.g. 'Acceptor Sequences')")
    args = parser.parse_args()

    create_plots(
        canonical_folder=args.canonical,
        noncanonical_folder=args.noncanonical,
        synthetic_folder=args.synthetic,
        title_prefix=args.title
    )

if __name__ == "__main__":
    main()
