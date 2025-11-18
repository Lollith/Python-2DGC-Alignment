#!/usr/bin/env python
import argparse
import os
import sys
import h5py
import netCDF4 as nc
from consensus_precompressor import PeakPrecompressor
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def setup_logging(output_path):
    """Configure logging pour le subprocess CLI avec flush automatique"""
    log_file = os.path.join(output_path, "precompress.log")
    
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
    print("Starting Peak Precompressor ...", flush=True)
    parser = argparse.ArgumentParser(description="Peak Precompressor")
    parser.add_argument("--input", required=True, nargs='+', help="Input files")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--rt1_penalty', type=int, default=1, help="RT1 penalty")
    parser.add_argument('--rt2_penalty', type=int, default=10, help="RT2 penalty")
    parser.add_argument('--similarity_cutoff', type=float, default=95.0, help="Similarity cutoff")
    parser.add_argument('--num_cores', type=int, default=1, help="Number of cores to use")
    # parser.add_argument('--common_ions', nargs='+', type=int, default=None, help="List of common ions")
    parser.add_argument('--quant_method', type=str, default="T", help="Quantification method")
    # parser.add_argument("--output", type=bool, required=True, help="")
    parser.add_argument('--area_selection', type=str, default="area_mod_max", help="Area selection method")
    args = parser.parse_args()
    logger = setup_logging(args.output_dir)
    try:
        precompressedFiles = PeakPrecompressor(
            rt1_penalty=args.rt1_penalty,
            rt2_penalty=args.rt2_penalty,
            similarity_cutoff=args.similarity_cutoff,
            num_cores=args.num_cores,
            # common_ions=args.common_ions,
            quant_method=args.quant_method,
            area_selection=args.area_selection
        )
        logger.info(f"\n{'='*60}")
        # logger.info(f"\n🔍 Starting analysis...")
        for i, f in enumerate(args.input, 1):
            if f.startswith(precompressedFiles.docker_volume_path):
                display_path = f.replace(precompressedFiles.docker_volume_path, "")
            else:
                display_path = f
            logger.info(f"  {i}. {display_path}")
        precompressedFiles.precompress_files(args.input, args.output_dir)
        logger.info("Precompress log saved: " + os.path.join(args.output_dir, "precompress.log"))
        # logger.info(f"✅ Analysis completed successfully!")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
