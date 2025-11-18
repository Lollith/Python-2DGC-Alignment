
import argparse
import os
import sys
from consensus_aligner import ChromatographicAligner
import logging
# from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def setup_logging(output_path):
    """Configure logging pour le subprocess CLI avec flush automatique"""
    log_file = os.path.join(output_path, "align.log")
    
    # Handler personnalisé avec flush automatique
    class FlushStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    console_handler = FlushStreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    return logging.getLogger('gcgc_cli')

def main():
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Alignment CLI")
    parser.add_argument("--input", required=True, nargs='+', help="Input files")
    parser.add_argument('--seed_file', type=int, required=True, help="Seed file index")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save output files")
    parser.add_argument('--rt1_penalty', type=float, default=1.0, help="RT1 penalty")
    parser.add_argument('--rt2_penalty', type=float, default=5.0, help="RT2 penalty")
    parser.add_argument('--similarity_cutoff', type=int, default=90, help="Similarity cutoff")
    parser.add_argument('--disimilarity_cutoff', type=int, default=90, help="Disimilarity cutoff")
    parser.add_argument('--num_cores', type=int, default=1, help="Number of cores to use")
    parser.add_argument('--missing_value_limit', type=float, default=0.05, help="Missing value limit")
    parser.add_argument('--quant_method', type=str, default="T", help="Quantification method")
    parser.add_argument('--missing_peak_finder_similarity_lax', type=float, default=0.85, help="Missing peak finder similarity lax")
    parser.add_argument('--auto_tune_match_stringency', action='store_true', help="Auto-tune match stringency")
    parser.add_argument('--nist', action='store_true', help="Enable NIST database matching")
    parser.add_argument('--area_selection', type=str, default="area_mod_max", help="Area selection method")
    args = parser.parse_args()

    logger = setup_logging(args.output_path)
    if not args.input:
        logger.error("❌ Error: No files selected for analysis.")
        return
    # print(f"📁 CLI received output_path: {args.output_path}", flush=True)
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
            missing_peak_finder_similarity_lax=args.missing_peak_finder_similarity_lax,
            output_path=args.output_path,
            area_selection=args.area_selection
        )
        aligner.consensus_align_bis(args.input, args.seed_file,
                                    common_ions=None,
                                    )
        aligner.filter_alignment_matrix()
        aligner.nist_identification(args.nist, match_factor_min=650)
        aligner.save_results()
        logger.info("Alignment log saved: " + os.path.join(args.output_path, "align.log"))
        # logger.info("✅ Alignment completed successfully.")
    except Exception as e:
        logger.error(f"Error during alignment: {e}")


if __name__ == "__main__":
    main()