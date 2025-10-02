import argparse
from consensus_aligner import ChromatographicAligner
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def main():
    print("Starting Peak Alignment Filter ...")
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Alignment with filter CLI")
    parser.add_argument("--path", required=True, nargs='+', help="Input files .csv from peak alignment" )
    parser.add_argument('--output_path', type=str, default=".", help="Output path")
    parser.add_argument('--new_missing_value_limit', type=float, default=0.05, help="Missing value limit")
    args = parser.parse_args()
    try:
        aligner = ChromatographicAligner(
            input_filter_path=args.path, #recupere directement mon output de l alignement comme input
            output_path=args.output_path,
        )
        threshold = float(args.new_missing_value_limit)
        print(f"Filtering alignment matrix with new missing value limit: {threshold}")
        aligner.load_csv_results()
        aligner.filter_alignment_matrix(threshold)
        aligner.save_results(with_filter=True)
    except Exception as e:
        print(f"Error during filtering: {e}")


if __name__ == "__main__":
    main()
