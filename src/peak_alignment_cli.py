
import argparse
import os
import sys
from consensus_aligner import ChromatographicAligner
# from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Alignment CLI")
    parser.add_argument("--input", required=True, nargs='+', help="Input files")
    parser.add_argument('--seed_file', type=int, required=True, help="Seed file index")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save output files")
    parser.add_argument('--rt1_penalty', type=int, default=1, help="RT1 penalty")
    parser.add_argument('--rt2_penalty', type=int, default=5, help="RT2 penalty")
    parser.add_argument('--similarity_cutoff', type=int, default=90, help="Similarity cutoff")
    parser.add_argument('--disimilarity_cutoff', type=int, default=90, help="Disimilarity cutoff")
    parser.add_argument('--num_cores', type=int, default=1, help="Number of cores to use")
    parser.add_argument('--missing_value_limit', type=float, default=0.05, help="Missing value limit")
    parser.add_argument('--quant_method', type=str, default="T", help="Quantification method")
    parser.add_argument('--missing_peak_finder_similarity_lax', type=float, default=0.85, help="Missing peak finder similarity lax")
    parser.add_argument('--auto_tune_match_stringency', action='store_true', help="Auto-tune match stringency")
    parser.add_argument('--nist', action='store_true', help="Enable NIST database matching")
    args = parser.parse_args()

    if not args.input:
        print("❌ Error: No files selected for analysis.")
        return
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
            missing_peak_finder_similarity_lax=args.missing_peak_finder_similarity_lax;
            output_path=args.output_path
        )
        aligner.consensus_align_bis(args.input, args.seed_file,
                                    common_ions=None,
                                    )
        aligner.filter_alignment_matrix()
        aligner.nist_identification(args.nist, match_factor_min=650 )
        aligner.save_results()
    except Exception as e:
        print(f"Error during alignment: {e}")


if __name__ == "__main__":
    main()