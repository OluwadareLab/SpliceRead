import argparse
from nucleotide_content_plotter import generate_plots

def main():
    parser = argparse.ArgumentParser(description="Plot nucleotide content scatter plots.")
    parser.add_argument('--canonical', required=True, help="Canonical sequence folder")
    parser.add_argument('--noncanonical', required=True, help="Non-canonical sequence folder")
    parser.add_argument('--synthetic', required=True, help="Synthetic sequence folder")
    parser.add_argument('--title', required=True, help="Title prefix for plots")
    parser.add_argument('--output', required=True, help="Directory to save plots")
    args = parser.parse_args()

    generate_plots(
        canonical_folder=args.canonical,
        noncanonical_folder=args.noncanonical,
        synthetic_folder=args.synthetic,
        title_prefix=args.title,
        save_directory=args.output
    )

if __name__ == "__main__":
    main()