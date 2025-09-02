#!/usr/bin/env python
import argparse
import os
import sys
import h5py
import netCDF4 as nc
from consensus_precompressor import PeakPrecompressor

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    parser = argparse.ArgumentParser(description="Peak Precompressor")
    parser.add_argument('--rt1_penalty', type=int, default=1, help="RT1 penalty")
    parser.add_argument('--rt2_penalty', type=int, default=10, help="RT2 penalty")
    parser.add_argument('--similarity_cutoff', type=float, default=95.0, help="Similarity cutoff")
    parser.add_argument('--num_cores', type=int, default=1, help="Number of cores to use")
    parser.add_argument('--common_ions', type=list, default=None, help="List of common ions")
    parser.add_argument('--quant_method', type=str, default="T", help="Quantification method")
    parser.add_argument("--output", type=str, required=True, help="")
    args = parser.parse_args()

    try:
        precompressedFiles = PeakPrecompressor(
            rt1_penalty=args.rt1_penalty,
            rt2_penalty=args.rt2_penalty,
            similarity_cutoff=args.similarity_cutoff,
            num_cores=args.num_cores,
            common_ions=args.common_ions,
            quant_method=args.quant_method,
            output_dir=args.output_dir
        )
        precompressedFiles.precompressFiles()
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
