#!/usr/bin/env python
import argparse
import os
import sys
import h5py
import netCDF4 as nc
from consensus_precompressor import PeakPrecompressor

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


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
        # print(f"🔍 Starting analysis of {len(args.input)} files: {args.input}...", flush=True)
        # print(f"\n{'='*60}", flush=True)
        for i, f in enumerate(args.input, 1):
            if f.startswith(precompressedFiles.docker_volume_path):
                display_path = f.replace(precompressedFiles.docker_volume_path, "")
            else:
                display_path = f
            print(f"  {i}. {display_path}")
        precompressedFiles.precompress_files(args.input, args.output_dir)

        print(f"✅ Analysis completed successfully!", flush=True)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
