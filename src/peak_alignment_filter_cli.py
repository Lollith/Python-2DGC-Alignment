import argparse
from consensus_aligner import ChromatographicAligner
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    print("Starting Peak Alignment Filter CLI...")
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Alignment with filter CLI")
    parser.add_argument("--input", required=True, nargs='+', help="Input files .cdf or .h5")
    parser.add_argument("----new_missing_value_limit", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument('--rt1_penalty', type=int, default=1, help="RT1 penalty")
    parser.add_argument('--rt2_penalty', type=int, default=5, help="RT2 penalty")
    parser.add_argument('--similarity_cutoff', type=int, default=90, help="Similarity cutoff")
    parser.add_argument('--disimilarity_cutoff', type=int, default=90, help="Disimilarity cutoff")
    parser.add_argument('--num_cores', type=int, default=1, help="Number of cores to use")
    parser.add_argument('--missing_value_limit', type=float, default=0.05, help="Missing value limit")
    parser.add_argument('--quant_method', type=str, default="T", help="Quantification method")
    parser.add_argument('--auto_tune_match_stringency', type=bool, default=False, help="Auto-tune match stringency")
    parser.add_argument('--missing_peak_finder_similarity_lax', type=float, default=0.85, help="Missing peak finder similarity lax")
    args = parser.parse_args()

    try:
        aligner = ChromatographicAligner(
            rt1_penalty=args.rt1_penalty,
            rt2_penalty=args.rt2_penalty,
            similarity_cutoff=args.similarity_cutoff,
            disimilarity_cutoff=args.disimilarity_cutoff,
            num_cores=args.num_cores,
            missing_value_limit=args.missing_value_limit,
            quant_method=args.quant_method,
            auto_tune_match_stringency=args.auto_tune_match_stringency,
            missing_peak_finder_similarity_lax=args.missing_peak_finder_similarity_lax
        )
        aligner.consensus_align_bis(
            input_file_list=args.input,
            seed_file=1,
        )
        threshold = float(args.new_missing_value_limit)
        aligner.filter_alignment_matrix(threshold)
        aligner.save_results(args.output_path, with_filter=True)
    except Exception as e:
        print(f"Error during filtering: {e}")