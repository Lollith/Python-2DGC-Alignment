#!/usr/bin/env python
import argparse
import os
import sys
import h5py
import netCDF4 as nc
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from identification import sample_identification

docker_volume_path = os.environ.get("DOCKER_VOLUME_PATH", "/app/data/")

def setup_logging(output_path):
    """Configure logging pour le subprocess CLI avec flush automatique"""
    log_file = os.path.join(output_path, "peak_detection.log")
    
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


def save_parameters(selected_files, output_path, method, mode, noise_factor,
                    min_persistence, abs_threshold, rel_threshold, cluster,
                    min_distance, min_sigma, max_sigma, sigma_ratio, num_sigma,
                    formated_spectra, match_factor_min, overlap, eps,
                    min_samples, nist):
    """Save the analysis parameters to a file."""
    params = {
        "selected_files": selected_files,
        "method": method,
        "mode": mode,
        "noise_factor": noise_factor,
        "min_persistence": min_persistence,
        "abs_threshold": abs_threshold,
        "rel_threshold": rel_threshold,
        "cluster": cluster,
        "min_distance": min_distance,
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "sigma_ratio": sigma_ratio,
        "num_sigma": num_sigma,
        "formated_spectra": formated_spectra,
        "match_factor_min": match_factor_min,
        "overlap": overlap,
        "eps": eps,
        "min_samples": min_samples,
        "nist": nist
    }
    with open(os.path.join(output_path, 'analysis.params'), 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def get_scan_number(file_path):
    """Get scan number from file."""
    try:
        if file_path.endswith((".h5", ".H5")):
            with h5py.File(file_path, 'r') as f:
                return f.attrs['scan_number_size']
        elif file_path.endswith((".cdf", ".CDF")):
            with nc.Dataset(file_path, 'r') as dt:
                return dt.dimensions['scan_number'].size
        else:
            raise ValueError("Unsupported file format. Please provide a .h5 or .cdf file.")
    except Exception as e:
        raise ValueError(f"Error while reading file {file_path}: {e}")


def get_mod_time(file_path):
    """Get modulation time based on scan_number from file."""
    scan_number = get_scan_number(file_path)
    modulation_times = {
        328125: (1.25, "G0/plasma"),
        540035: (1.7, "exhaled air")
    }
    if scan_number in modulation_times:
        mod_time, data_type = modulation_times[scan_number]
        logging.info(f"   Data type: {data_type}")
        return mod_time
    else:
        logging.warning(f"   ⚠️  Unknown scan_number: {scan_number}, using default modulation time")
        return


def main():
    parser = argparse.ArgumentParser(description="GC×GC-MS Peak Detection CLI")
    parser.add_argument("--input", required=True, nargs='+', help="Input files .cdf or .h5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--noise_factor", type=float, required=True)
    parser.add_argument("--min_persistence", type=float, required=True)
    parser.add_argument("--abs_threshold", type=float, required=True)
    parser.add_argument("--rel_threshold", type=float, required=True)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--min_distance", type=int, required=True)
    parser.add_argument("--min_sigma", type=float, required=True)
    parser.add_argument("--max_sigma", type=float, required=True)
    parser.add_argument("--sigma_ratio", type=float, required=True)
    parser.add_argument("--num_sigma", type=int, required=True)
    parser.add_argument("--formated_spectra", action="store_true")
    parser.add_argument("--match_factor_min", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--min_samples", type=int, required=True)
    parser.add_argument("--nist", action="store_true")
    parser.add_argument("--mod_time", type=float, help="Manual modulation time in seconds (0 to auto-detect)", default=0)
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--is_area_deconvolution", action="store_true", help="Use area from deconvolution in addition to mod_max")
    args = parser.parse_args()
    
    logger = setup_logging(args.output)
    if not args.input:
        logger.error("❌ Error: No files selected for analysis.")
        return
    save_parameters(
            args.input, args.output, args.method, args.mode, args.noise_factor,
            args.min_persistence, args.abs_threshold, args.rel_threshold,
            args.cluster, args.min_distance, args.min_sigma, args.max_sigma,
            args.sigma_ratio, args.num_sigma, args.formated_spectra,
            args.match_factor_min, args.overlap, args.eps, args.min_samples,
            args.nist
        )
    successful_analyses = 0
    failed_analyses = 0
    for i, f in enumerate(args.input, 1):
        if f.startswith(docker_volume_path):
            display_path = f.replace(docker_volume_path, "")
        else:
            display_path = f
        logger.info(f"  {i}. {display_path}")
    logger.info(f"\n{'='*60}")
    logger.info("\n🔍 Starting analysis...")

    for i, f in enumerate(args.input, 1):
        if f.startswith(docker_volume_path):
            display_path = f.replace(docker_volume_path, "")
        else:
            display_path = f
        logger.info(f"\n{i}. {display_path}")
        try:
            path = os.path.dirname(f)
            file = os.path.basename(f)

            if args.mod_time and args.mod_time > 0:
                mod_time = args.mod_time
                logger.info(f"⏱️  Using manual modulation time: {mod_time} seconds")
            else:
                mod_time = get_mod_time(f)
                if mod_time is None:
                    logger.warning("   ⚠️ Modulation time not specified, using default value of 1.25 seconds")
                    mod_time = 1.25
                logger.info(f"⏱️  Modulation time: {mod_time} seconds")

            results = sample_identification(
                path,
                file,
                args.output,
                mod_time,
                args.method, args.mode,
                args.noise_factor,
                args.abs_threshold,
                args.rel_threshold,
                args.cluster,
                args.min_distance,
                args.min_sigma,
                args.max_sigma,
                args.sigma_ratio,
                args.num_sigma,
                args.formated_spectra,
                args.match_factor_min,
                args.min_persistence,
                args.overlap,
                args.eps,
                args.min_samples,
                args.nist,
                args.plot,
                args.is_area_deconvolution
            )
            if isinstance(results, str) and (results.startswith("❌") or results.startswith("⚠️")):
                logger.error(results)  # Afficher le message d'erreur
                failed_analyses += 1
            elif isinstance(results, list):
            # affichage du path
                if args.output.startswith(docker_volume_path):
                    display_path = args.output.replace(docker_volume_path, "")
                else:
                    display_path = args.output
                logger.info(f"📂 Parameters saved to '{display_path}analysis.params'")

                for result in results:
                    if result.startswith(docker_volume_path):
                        display_path = result.replace(docker_volume_path, "")
                    else:
                        display_path = result
                    logger.info(f"✅ Fichier {file} traité, résultats: {display_path}")
                successful_analyses += 1
                logger.info("✅ Analysis completed successfully!")
            else:
                logger.info(results)
                successful_analyses += 1
        except Exception as e:
            logger.error(f"❌ Analysis failed for {f}:")
            logger.error(f"   Error: {str(e)}")
            failed_analyses += 1
            
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ANALYSIS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Réussies: {successful_analyses}")
        logger.info(f"❌ Échouées: {failed_analyses}")
        logger.info(f"📈 Taux de succès: {successful_analyses}/{len(args.input)} ({100*successful_analyses/len(args.input):.1f}%)")
        logger.info("Peak Detection log saved: " + os.path.join(args.output, "peak_detection.log"))
        logger.info(f"\n{'-'*60}")


if __name__ == "__main__":
    main()
