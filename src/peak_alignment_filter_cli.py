import argparse

from sklearn import logger
from consensus_aligner import ChromatographicAligner
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def setup_logging(output_path):
    """Configure logging pour le subprocess CLI avec flush automatique"""
    log_file = os.path.join(output_path, "filter.log")
    
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
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Alignment with filter CLI")
    parser.add_argument("--path", required=True, nargs='+', help="Input files .csv from peak alignment" )
    parser.add_argument('--output_path', type=str, default=".", help="Output path")
    parser.add_argument('--new_missing_value_limit', type=float, default=0.05, help="Missing value limit")
    args = parser.parse_args()
    logger = setup_logging(args.output_path)
    try:
        aligner = ChromatographicAligner(
            input_filter_path=args.path, #recupere directement mon output de l alignement comme input
            output_path=args.output_path,
        )
        threshold = float(args.new_missing_value_limit)
        # affichage du path
        input_dir = os.path.dirname(args.path[0])
        if input_dir.startswith(aligner.docker_volume_path):
            display_path = input_dir.replace(aligner.docker_volume_path, "")
        else:
            display_path = input_dir
        logger.info(f"Loaded alignment matrix from: {display_path}/")

        logger.info(f"Filtering alignment matrix with new missing value limit: {threshold}")
        aligner.load_csv_results()
        aligner.filter_alignment_matrix(threshold)
        aligner.save_results(with_filter=True)
        logger.info("Filter log saved: " + os.path.join(args.output_path, "filter.log"))
        # logger.info("✅ Filtering completed successfully.")
    except Exception as e:
        logger.error(f"Error during filtering: {e}")


if __name__ == "__main__":
    main()
