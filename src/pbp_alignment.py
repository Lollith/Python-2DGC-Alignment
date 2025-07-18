# import gc
# import json
# import os
# import time
# import warnings
# from concurrent.futures import ProcessPoolExecutor

# import numpy as np
# from netCDF4 import Dataset
# from ngl import natgrid
# from scipy.interpolate import interp1d, interpn

# from swpa_peak_alignment import swpa_peak_alignment


# def load_config(config_path):
#     """
#     Loads the configuration parameters from the specified JSON file.

#     Parameters:
#         config_path (str): Path to the configuration file.

#     Returns:
#         dict: A dictionary containing the configuration parameters.
#     """
#     with open(config_path, "r") as config_file:
#         return json.load(config_file)


# def open_chromatogram(filename, int_thresh, drift_ms):
#     """
#     Opens a chromatogram file and extracts the relevant data.

#     Parameters:
#         filename (str): Path to the chromatogram file.
#         int_thresh (int): Intensity threshold for filtering.
#         drift_ms (float): Drift time in milliseconds.

#     Returns:
#         dict: A dictionary containing the chromatogram data
#     """
#     # Open the CDF file
#     with Dataset(filename, "r") as cdf:
#         # Retrieve data from the file
#         flag = cdf.variables["flag_count"][:]
#         scan_time = cdf.variables["scan_acquisition_time"][:].astype(np.float32)
#         scan_num = cdf.variables["actual_scan_number"][:]
#         med_ms_max = cdf.variables["mass_range_max"][:].astype(np.float32)
#         med_ms_min = cdf.variables["mass_range_min"][:].astype(np.float32)
#         ion_id = cdf.variables["scan_index"][:]
#         each_scan_num = cdf.variables["point_count"][:]
#         ms_tot_int = cdf.variables["total_intensity"][:].astype(np.float32)
#         ms_value = (cdf.variables["mass_values"][:] + drift_ms).astype(np.float32)
#         ms_int = cdf.variables["intensity_values"][:].astype(np.float32)

#     # Compute Time parameters
#     time_para = scan_time[np.abs(each_scan_num) < np.iinfo(np.int32).max]
#     rt_ini = np.min(time_para) / 60
#     rt_runtime = np.max(time_para) / 60 - rt_ini
#     sam_rate = 1 / np.mean(time_para[1:] - time_para[:-1])

#     # Initialize data arrays
#     pixel_num = len(scan_num)
#     max_scan_num = np.max(each_scan_num)
#     ms_value_box = np.zeros((pixel_num, max_scan_num + 1))
#     ms_int_box = np.zeros((pixel_num, max_scan_num + 1))

#     # Populate MS data boxes
#     initials = np.cumsum(each_scan_num) - each_scan_num
#     acq_ranges = [
#         np.arange(
#             min(initials[i], initials[i] + each_scan_num[i] - 1),
#             max(initials[i], initials[i] + each_scan_num[i] - 1) + 1,
#         )
#         if each_scan_num[i] != 0
#         else []
#         for i in range(pixel_num)
#     ]

#     for i_num in range(pixel_num):
#         acq_range = acq_ranges[i_num]
#         ms_value_box[i_num, : len(acq_range)] = ms_value[acq_range]
#         ms_int_box[i_num, : len(acq_range)] = ms_int[acq_range]

#     # Apply intensity threshold
#     ms_value_box[ms_int_box < int_thresh] = 0
#     ms_int_box[ms_int_box < int_thresh] = 0

#     # Organize output into a dictionary
#     chromato = {
#         "flag": flag,
#         "scantime": scan_time,
#         "scannum": scan_num,
#         "medmsmax": med_ms_max,
#         "medmsmin": med_ms_min,
#         "eachscannum": each_scan_num,
#         "MStotint": ms_tot_int,
#         "MSvalue": ms_value,
#         "MSint": ms_int,
#         "Timepara": time_para,
#         "RTini": rt_ini,
#         "RTruntime": rt_runtime,
#         "SamRate": sam_rate,
#         "MSthreshold": int_thresh,
#         "pixelnum": pixel_num,
#         "maxscannum": max_scan_num,
#         "minscannum": np.min(each_scan_num),
#         "MSvaluebox": ms_value_box,
#         "MSintbox": ms_int_box,
#         "ionid": ion_id,
#     }

#     return chromato


# def load_chromatograms(input_path, target_file, reference_file, int_thresh, drift_ms):
#     """
#     Loads the target and reference chromatograms from the input path.

#     Parameters:
#         input_path (str): Path to the input directory.
#         target_file (str): Name of the target chromatogram file.
#         reference_file (str): Name of the reference chromatogram file.
#         int_thresh (int): Intensity threshold for filtering.
#         drift_ms (float): Drift time in milliseconds.

#     Returns:
#         tuple: Two dictionaries containing the target and reference chromatograms.
#     """
#     target_chromato_path = os.path.join(input_path, target_file)
#     reference_chromato_path = os.path.join(input_path, reference_file)

#     if not os.path.exists(target_chromato_path):
#         raise FileNotFoundError(
#             f"Target chromatogram file {target_chromato_path} does not exist."
#         )
#     if not os.path.exists(reference_chromato_path):
#         raise FileNotFoundError(
#             f"Reference chromatogram file {reference_chromato_path} does not exist."
#         )

#     chromato_target = open_chromatogram(
#         target_chromato_path, int_thresh=int_thresh, drift_ms=drift_ms
#     )
#     chromato_ref = open_chromatogram(
#         reference_chromato_path, int_thresh=int_thresh, drift_ms=drift_ms
#     )

#     return chromato_target, chromato_ref


# def time_to_pix(rttime, mod_time, freq, isot=0):
#     """
#     Converts retention times from units of time to units of pixels.

#     Parameters:
#     - rttime: numpy array of shape (n, 2), where each row is [retention time 1, retention time 2]
#     - mod_time: Modulation rate, in seconds
#     - freq: Sampling frequency, in Hz
#     - isot: Optional suppression time at the beginning of the chromatogram, in minutes (default is 0)

#     Returns:
#     - rtpix: numpy array of shape (n, 2), where each row corresponds to [pixel 1, pixel 2]
#     """
#     # Ensure `isot` is in minutes if not provided
#     rtpix = np.zeros_like(rttime, dtype=int)
#     # Calculate pixels based on time values
#     rtpix[:, 1] = np.round(rttime[:, 1] * freq).astype(int)
#     rtpix[:, 0] = np.round((rttime[:, 0] - isot) * 60 / mod_time).astype(int)

#     return rtpix


# def load_alignment_points(chromato_ref, chromato_target, config):
#     """
#     Imports and processes alignment points for reference and target chromatograms.

#     Parameters:
#     - chromato_ref: chromatogram object with `SamRate` and `RTini` attributes for the reference
#     - chromato_target: chromatogram object with `SamRate` and `RTini` attributes for the target
#     - config: dictionary containing configuration parameters

#     Returns:
#     - reference_peaks, target_peaks, typical_peak_width: numpy arrays
#     """

#     reference_file = config["io_params"]["REFERENCE_ALIGNMENT_PTS_FILE"]
#     target_file = config["io_params"]["TARGET_ALIGNMENT_PTS_FILE"]
#     input_path = config["io_params"]["INPUT_PATH"]
#     output_path = config["io_params"]["OUTPUT_PATH"]
#     units = config["model_choice_params"]["UNITS"]
#     mod_time = config["instrument_params"]["MODTIME"]
#     typical_peak_width = config["model_choice_params"]["TYPICAL_PEAK_WIDTH"]

#     # Helper function to find and load a file
#     def load_file(filename, input_path, output_path):
#         input_full_path = os.path.join(input_path, filename)
#         output_full_path = os.path.join(output_path, filename)

#         # Check if the file exists in the input or output path
#         if os.path.exists(input_full_path):
#             data = np.genfromtxt(input_full_path, delimiter=",")
#         elif os.path.exists(output_full_path):
#             warnings.warn(
#                 f"The file {filename} was found in the output_path, not in input_path. Loading from output_path."
#             )
#             data = np.genfromtxt(output_full_path, delimiter=",")
#         else:
#             raise FileNotFoundError(
#                 f"The file {filename} does not exist in either input or output paths."
#             )

#         # Ensure data has two columns and return
#         if data.shape[1] == 2:
#             return data
#         else:
#             raise ValueError(
#                 f"The file {filename} does not have 2 columns as expected."
#             )

#     # Load reference and target peaks
#     reference_peaks = load_file(reference_file, input_path, output_path)
#     target_peaks = load_file(target_file, input_path, output_path)

#     if units.lower() == "time":
#         # Convert time units to pixel units
#         target_peaks = time_to_pix(
#             target_peaks,
#             mod_time,
#             chromato_target["SamRate"],
#             chromato_target["RTini"],
#         )
#         reference_peaks = time_to_pix(
#             reference_peaks,
#             mod_time,
#             chromato_ref["SamRate"],
#             chromato_ref["RTini"],
#         )
#         typical_peak_width = time_to_pix(
#             np.array([typical_peak_width]),
#             mod_time,
#             chromato_ref["SamRate"],
#             chromato_ref["RTini"],
#         )
#         config["model_choice_params"]["TYPICAL_PEAK_WIDTH"] = typical_peak_width[0]

#     return reference_peaks.astype(int), target_peaks.astype(int)


# def reshape_tic(ms_totint, nb_pix_2nd_d):
#     """
#     Reshapes the ms_totint array into a 2D array with rows of size nb_pix_2nd_d,
#     padding with zeros if necessary.

#     Parameters:
#         ms_totint (numpy.ndarray): Input 1D array of intensities.
#         nb_pix_2nd_d (int): Number of rows for the reshaped array.

#     Returns:
#         numpy.ndarray: Reshaped 2D array.
#     """
#     # Calculate the padding length to make the array divisible by nb_pix_2nd_d
#     pad_length = nb_pix_2nd_d - (len(ms_totint) % nb_pix_2nd_d)
#     if pad_length == nb_pix_2nd_d:  # No padding needed
#         pad_length = 0

#     padded_ms_totint = np.pad(ms_totint, (0, pad_length), mode="constant")

#     # Reshape the padded array
#     return np.reshape(padded_ms_totint, (nb_pix_2nd_d, -1), order="F")


# def aggregate_unique_ms_data(mz_values, intensities):
#     """
#     Aggregate intensity values corresponding to unique m/z (mass-to-charge) values for each row in a matrix.

#     Parameters:
#     mz_values (numpy.ndarray): A 2D array where each row contains m/z values for a spectrum.
#     intensities (numpy.ndarray): A 2D array where each row contains intensity values corresponding to the m/z values.

#     Returns:
#     tuple: Two numpy arrays:
#         - final_mz_values: 2D array with unique m/z values for each row.
#         - final_intensities: 2D array with aggregated intensity values corresponding to the unique m/z values.
#     """
#     unique_mz_values = np.zeros_like(mz_values)
#     aggregated_intensities = np.zeros_like(mz_values)

#     for kt in range(mz_values.shape[0]):
#         # Get unique m/z values and their indices
#         unique_mz, indices = np.unique(mz_values[kt, :], return_inverse=True)

#         # Aggregate corresponding intensities
#         aggregated_values = np.zeros_like(unique_mz, dtype=intensities.dtype)
#         np.add.at(aggregated_values, indices, intensities[kt, :])

#         # Update results in the final arrays
#         unique_mz_values[kt, : len(unique_mz)] = unique_mz
#         aggregated_intensities[kt, : len(unique_mz)] = aggregated_values

#     # Remove unnecessary trailing zeros
#     max_nonzero_columns = np.max(np.sum(unique_mz_values != 0, axis=1))
#     final_mz_values = unique_mz_values[:, :max_nonzero_columns]
#     final_intensities = aggregated_intensities[:, :max_nonzero_columns]

#     return final_mz_values, final_intensities


# def round_and_aggregate_unique_ms_data(mz_values, intensities, precision):
#     """
#     Round m/z values to a specified precision and aggregate corresponding intensity values to unique m/z values.

#     Parameters:
#     mz_values (numpy.ndarray): A 2D array where each row contains m/z values for a spectrum.
#     intensities (numpy.ndarray): A 2D array where each row contains intensity values corresponding to the m/z values.
#     precision (float): The precision to which the m/z values should be rounded (e.g., 0.001).

#     Returns:
#     tuple: Two numpy arrays:
#         - rounded_mz_values: 2D array with unique rounded m/z values for each row.
#         - aggregated_intensities: 2D array with aggregated intensity values corresponding to the rounded m/z values.
#     """
#     rounded_mz_values = np.around(mz_values, decimals=precision)
#     return aggregate_unique_ms_data(rounded_mz_values, intensities)


# def align_2d_chrom_ms_v5(
#     ref, other, peaks_ref, peaks_other, ms_valuebox, ms_intbox, nb_pix_2nd_d, **kwargs
# ):
#     """
#     Aligns 2D chromatograms with MS data.

#     Parameters:
#         ref (np.ndarray): Reference chromatogram of size [m, n].
#         other (np.ndarray): Target chromatogram of size [m, n].
#         peaks_ref (np.ndarray): Alignment points in the reference chromatogram.
#         peaks_other (np.ndarray): Alignment points in the target chromatogram.
#         ms_valuebox (np.ndarray): m/z values of ions.
#         ms_intbox (np.ndarray): Corresponding intensity values of ions.
#         nb_pix_2nd_d (int): Number of pixels per 2nd dimension column.
#         kwargs (dict): Optional arguments:
#             - 'power_factor' (float): Weighting factor for deformation correction.
#             - 'peak_widths' (list): Expected widths of peaks.
#             - 'inter_pixel_interp_meth' (str): Method for inter-pixel interpolation.
#             - 'model_choice' (str): Choice of model for alignment: 'normal' or 'DualSibson'.

#     Returns:
#         dict: A dictionary containing alignment results:
#             - AlignedMSvaluebox (np.ndarray)
#             - AlignedMSintbox (np.ndarray)
#             - Alignedeachscannum (np.ndarray)
#             - Alignedmedionid (np.ndarray)
#             - Aligned (np.ndarray)
#             - Displacement (np.ndarray)
#             - Deform_output (np.ndarray)
#     """

#     # Default values for optional arguments
#     power_factor = kwargs.get("power_factor", 2.0)
#     peak_widths = kwargs.get("peak_widths", [1, 1])
#     inter_pixel_interp_meth = kwargs.get("inter_pixel_interp_meth", "linear")
#     model_choice = kwargs.get("model_choice", "normal")

#     if model_choice not in ["normal", "DualSibson"]:
#         raise ValueError(f"Invalid model choice: {model_choice}.")

#     aligned, displacement, deform_output = align_chromato(
#         ref=ref,
#         target=np.squeeze(other),
#         peaks_ref=peaks_ref,
#         peaks_target=peaks_other,
#         model_choice=model_choice,
#         power_factor=power_factor,
#         peak_widths=peak_widths,
#         inter_pixel_interp_meth=inter_pixel_interp_meth,
#     )

#     aligned_msvaluebox_i, aligned_msintbox_i = bilinear_interpolation_alignment(
#         ms_valuebox, ms_intbox, displacement, deform_output, nb_pix_2nd_d
#     )

#     aligned_msvaluebox_ii, aligned_msintbox_ii = sort_and_trim_nonzero_columns(
#         aligned_msvaluebox_i, aligned_msintbox_i
#     )
#     del aligned_msvaluebox_i, aligned_msintbox_i
#     gc.collect()

#     # -- Step 3: For each pixel, only keep each m/z value once, summing corresponding intensity values
#     # Initialize matrices for aggregated m/z values and intensities
#     final_mz_values, final_intensities = aggregate_unique_ms_data(
#         aligned_msvaluebox_ii, aligned_msintbox_ii
#     )
#     del aligned_msvaluebox_ii, aligned_msintbox_ii
#     gc.collect()

#     aligned_each_scan_num = np.full(
#         (final_intensities.shape[0], 1), final_intensities.shape[1]
#     )
#     aligned_medion_id = np.cumsum(aligned_each_scan_num)

#     return {
#         "MSvaluebox": final_mz_values,
#         "MSintbox": final_intensities,
#         "eachscannum": aligned_each_scan_num,
#         "ionid": aligned_medion_id,
#         "Aligned": aligned,
#         "Displacement": displacement,
#         "Deform_output": deform_output,
#     }


# def bilinear_interpolation_alignment(
#     ms_valuebox, ms_intbox, displacement, deform_output, nb_pix_2nd_d
# ):
#     vb_rows, vb_cols = ms_valuebox.shape
#     ib_rows, ib_cols = ms_intbox.shape
#     aligned_indices = np.zeros(vb_rows, dtype=np.float32)
#     aligned_msvaluebox = np.zeros((vb_rows, vb_cols * 4))
#     aligned_msintbox = np.zeros((ib_rows, ib_cols * 4))
#     interp_distr = np.zeros(vb_rows, dtype=np.float32)
#     interp_dists = np.zeros(vb_rows, dtype=np.float32)
#     interp_distt = np.zeros(vb_rows, dtype=np.float32)
#     interp_distu = np.zeros(vb_rows, dtype=np.float32)
#     defm = np.zeros(vb_rows, dtype=np.float32)

#     # -------------------------
#     frst_d_flag, scnd_d_flag = 0, 0
#     for ht in range(vb_rows):
#         # Compute r, s, t, u for interpolation
#         interp_distr[ht] = displacement[scnd_d_flag, frst_d_flag, 1] - np.floor(
#             displacement[scnd_d_flag, frst_d_flag, 1]
#         )
#         interp_dists[ht] = -displacement[scnd_d_flag, frst_d_flag, 1] + np.ceil(
#             displacement[scnd_d_flag, frst_d_flag, 1]
#         )
#         interp_distt[ht] = displacement[scnd_d_flag, frst_d_flag, 0] - np.floor(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         )
#         interp_distu[ht] = -displacement[scnd_d_flag, frst_d_flag, 0] + np.ceil(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         )

#         # Update pixel counts
#         if (ht + 1) % nb_pix_2nd_d != 0:
#             scnd_d_flag += 1
#         else:
#             frst_d_flag += 1
#             scnd_d_flag = 0

#     # Correct indices that are outside the chromatogram
#     interp_distr[interp_distr == 0] = 0.5
#     interp_dists[interp_dists == 0] = 0.5
#     interp_distt[interp_distt == 0] = 0.5
#     interp_distu[interp_distu == 0] = 0.5
#     # -------------------------

#     # Process for each corner (a, b, c, d)
#     for corner in range(4):
#         frst_d_flag, scnd_d_flag = 0, 0
#         aligned_indices.fill(0)
#         defm.fill(0)
#         msvaluebox_aligned = aligned_msvaluebox[
#             :, vb_cols * corner : vb_cols * (corner + 1)
#         ]
#         msintbox_aligned = aligned_msintbox[
#             :, ib_cols * corner : ib_cols * (corner + 1)
#         ]

#         for ht in range(vb_rows):
#             aligned_indices[ht], defm[ht] = compute_alignment(
#                 ht,
#                 displacement,
#                 nb_pix_2nd_d,
#                 scnd_d_flag,
#                 frst_d_flag,
#                 deform_output,
#                 corner,
#             )

#             # Update pixel counts
#             if (ht + 1) % nb_pix_2nd_d != 0:
#                 scnd_d_flag += 1
#             else:
#                 frst_d_flag += 1
#                 scnd_d_flag = 0

#         # Correct indices and handle out-of-bounds
#         aligned_indices = np.where(
#             (aligned_indices > np.max(vb_rows - 1)) | (aligned_indices < 0),
#             np.nan,
#             aligned_indices,
#         )

#         # Populate MS values and intensities
#         lp_inds = np.arange(len(aligned_indices))
#         lp_inds2 = lp_inds[~np.isnan(aligned_indices)]

#         for ht in lp_inds2:
#             msvaluebox_aligned[ht, :] = ms_valuebox[int(aligned_indices[ht]), :]
#             msintbox_aligned[ht, :] = ms_intbox[
#                 int(aligned_indices[ht]), :
#             ] * compute_intensities_factor(
#                 corner, ht, interp_distr, interp_dists, interp_distt, interp_distu, defm
#             )

#         msvaluebox_aligned[np.isnan(msvaluebox_aligned)] = 0
#         msintbox_aligned[np.isnan(msintbox_aligned)] = 0

#     return aligned_msvaluebox, aligned_msintbox


# def sort_and_trim_nonzero_columns(aligned_msvaluebox_i, aligned_msintbox_i):
#     """
#     Sorts the columns of the aligned MS value and intensity matrices in ascending order of m/z values.
#     Removes trailing zeros from the matrices.
#     """
#     # Remove zeros temporarily by replacing them with the maximum integer value
#     aligned_msvaluebox_i[aligned_msvaluebox_i == 0] = np.iinfo(np.int32).max

#     # Sort the values in ascending order (big values corresponding to zeros go at the end)
#     ix = np.argsort(aligned_msvaluebox_i, axis=1, kind="stable")
#     aligned_msvaluebox_i.sort(axis=1)

#     # Re-put zeros instead of the big values
#     aligned_msvaluebox_i[aligned_msvaluebox_i == np.iinfo(np.int32).max] = 0

#     # Put the values in aligned_msintbox_i in the corresponding order
#     aligned_msintbox_i = np.take_along_axis(aligned_msintbox_i, ix, axis=1)

#     # Find the max of non-zero values (to remove useless zeros)
#     max_not_zero = np.max(np.sum(aligned_msintbox_i != 0, axis=1))

#     # Remove useless zeros
#     aligned_msvaluebox_ii = aligned_msvaluebox_i[:, :max_not_zero]
#     aligned_msintbox_ii = aligned_msintbox_i[:, :max_not_zero]

#     return aligned_msvaluebox_ii, aligned_msintbox_ii


# def compute_alignment(
#     ht, displacement, nb_pix_2nd_d, scnd_d_flag, frst_d_flag, deform_output, corner
# ):
#     """
#     Computes the alignment for a given corner (a, b, c, or d).
#     """
#     if corner == 0:  # Corner a (floor-floor)
#         aligned_index = np.floor(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         ) + nb_pix_2nd_d * np.floor(displacement[scnd_d_flag, frst_d_flag, 1])
#     elif corner == 1:  # Corner b (ceil-floor)
#         aligned_index = np.floor(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         ) + nb_pix_2nd_d * np.ceil(displacement[scnd_d_flag, frst_d_flag, 1])
#     elif corner == 2:  # Corner c (floor-ceil)
#         aligned_index = np.ceil(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         ) + nb_pix_2nd_d * np.floor(displacement[scnd_d_flag, frst_d_flag, 1])
#     else:  # Corner d (ceil-ceil)
#         aligned_index = np.ceil(
#             displacement[scnd_d_flag, frst_d_flag, 0]
#         ) + nb_pix_2nd_d * np.ceil(displacement[scnd_d_flag, frst_d_flag, 1])

#     defm = (
#         deform_output[scnd_d_flag, frst_d_flag, 0]
#         * deform_output[scnd_d_flag, frst_d_flag, 1]
#         / 4
#     )
#     return aligned_index + ht, defm


# def compute_intensities_factor(
#     corner, ht, interp_distr, interp_dists, interp_distt, interp_distu, defm
# ):
#     """
#     Computes the intensity factor for a given corner (a, b, c, or d).
#     """
#     if corner == 0:
#         return interp_dists[ht] * interp_distu[ht] * defm[ht]
#     elif corner == 1:
#         return interp_distr[ht] * interp_distu[ht] * defm[ht]
#     elif corner == 2:
#         return interp_dists[ht] * interp_distt[ht] * defm[ht]
#     else:
#         return interp_distr[ht] * interp_distt[ht] * defm[ht]


# def align_chromato(ref, target, peaks_ref, peaks_target, model_choice, **kwargs):
#     """
#     Aligns the target chromatogram to the reference chromatogram.

#     Parameters:
#         ref (numpy.ndarray): Reference chromatogram (2D matrix).
#         target (numpy.ndarray): Target chromatogram (2D matrix).
#         peaks_ref (numpy.ndarray): Positions of alignment points in ref (Nx2 matrix).
#         peaks_target (numpy.ndarray): Positions of alignment points in target (Nx2 matrix).
#         model_choice (str): Choice of model for alignment: 'normal' or 'DualSibson'.
#         **kwargs: Optional parameters:
#             - 'power_factor' (float): Weighting factor for inverse distance, default 2.
#             - 'peak_widths' (list): Typical peak widths [width_1st_dim, width_2nd_dim], default [1, 1].
#             - 'inter_pixel_interp_meth' (str): Interpolation method for intensity, default 'cubic'.

#     Returns:
#         aligned (numpy.ndarray): Aligned chromatogram.
#         displacement (numpy.ndarray): Displacement matrix [m, n, 2].
#         deform_output (numpy.ndarray): Deformation correction matrix [m, n, 2].
#     """
#     # Default optional arguments
#     power_factor = kwargs.get("power_factor", 2)
#     peak_widths = kwargs.get("peak_widths", [1, 1])
#     inter_pixel_interp_meth = kwargs.get("inter_pixel_interp_meth", "cubic")

#     # Ensure ref and target have the same size
#     ref, target = equalize_size(ref, target)

#     # Compute the peak ratio to normalize distances
#     peak_ratio = peak_widths[1] / peak_widths[0]

#     # Compute displacement
#     peaks_displacement = peaks_target - peaks_ref

#     # Initialize output arrays
#     aligned = np.zeros_like(target)
#     displacement = np.zeros((target.shape[0], target.shape[1], 2), dtype=np.float32)

#     # -- Compute displacement of peaks and interpolate (1st dim: linear, 2nd dim: natural-neighbor)
#     displacement2 = np.zeros((2, 2, 2))

#     padding_w_lower = np.floor(0.05 * aligned.shape[0] + 2)
#     padding_w_upper = np.ceil(0.05 * aligned.shape[0] + 1)
#     padding_x_lower = np.floor(0.05 * aligned.shape[1] + 2)
#     padding_x_upper = np.ceil(0.05 * aligned.shape[1] + 1)

#     # Compute displacement2 based on the alignment and reference peaks
#     # w: 2nd dimension, x: 1st dimension
#     for w in (-padding_w_lower, aligned.shape[0] + padding_w_upper):
#         for x in (-padding_x_lower, aligned.shape[1] + padding_x_upper):
#             # Compute the distance vector for the pixel
#             distance_vec = np.array([w, x]) - np.flip(peaks_ref, axis=1)
#             distance = np.sqrt(
#                 distance_vec[:, 0] ** 2 + (distance_vec[:, 1] * peak_ratio) ** 2
#             )

#             # Compute the displacement using a weighted mean
#             weight = 1 / (distance**power_factor)
#             d2w = (w + int(padding_w_lower)) // (
#                 aligned.shape[0] + int(padding_w_lower) + int(padding_w_upper)
#             )
#             d2x = (x + int(padding_x_lower)) // (
#                 aligned.shape[1] + int(padding_x_lower) + int(padding_x_upper)
#             )
#             displacement2[int(d2w), int(d2x), :] = np.sum(
#                 np.flip(peaks_displacement, axis=1) * (weight[:, np.newaxis]), axis=0
#             ) / np.sum(weight)

#     # Augment the peaks with corner points to improve interpolation near edges.
#     peaks_ref_corner = np.vstack(
#         [
#             peaks_ref,
#             # base corners with offsets
#             np.array(
#                 [
#                     [-padding_x_lower, -padding_w_lower],
#                     [-padding_x_lower, aligned.shape[0] + padding_w_upper],
#                     [aligned.shape[1] + padding_x_upper, -padding_w_lower],
#                     [
#                         aligned.shape[1] + padding_x_upper,
#                         aligned.shape[0] + padding_w_upper,
#                     ],
#                 ]
#             ),
#         ]
#     )

#     # Add corresponding displacement values for the added peaks
#     peaks_displacement_corner = np.vstack(
#         [
#             peaks_displacement,
#             np.array(
#                 [
#                     [displacement2[0, 0, 1], displacement2[0, 0, 0]],
#                     [displacement2[1, 0, 1], displacement2[1, 0, 0]],
#                     [displacement2[0, 1, 1], displacement2[0, 1, 0]],
#                     [displacement2[1, 1, 1], displacement2[1, 1, 0]],
#                 ]
#             ),
#         ]
#     )

#     # Use Sibson interpolation (natgrid) for smooth interpolation for the 2nd dimension.
#     natw = peaks_ref_corner[:, 1]
#     natx = peaks_ref_corner[:, 0] * peak_ratio
#     wo = np.arange(-padding_w_lower, aligned.shape[0] + padding_w_upper)
#     xo = np.arange(
#         np.round(-padding_x_lower * peak_ratio),
#         np.round((aligned.shape[1] + padding_x_upper) * peak_ratio),
#     )

#     fdist1 = natgrid(natw, natx, peaks_displacement_corner[:, 1], wo, xo)
#     fdist2 = (
#         natgrid(natw, natx, peaks_displacement_corner[:, 0], wo, xo)
#         if model_choice == "DualSibson"
#         else None
#     )
    
#     def update_displacements(peaks_ref, peaks_displacement):
#         """
#         Update displacement values based on the reference peaks.

#         Args:
#         - peaks_ref (np.ndarray): Array of reference peaks with shape (n, 2), where each row represents [x, w].
#         - peaks_displacement (np.ndarray): Array of displacement values with shape (n, 2), where each row represents [displacement_x, displacement_w].

#         Returns:
#         - np.ndarray: Updated `peaks_displacement` array with the new displacement values.
#         """
#         ref_peaks = peaks_ref[:, 0]
#         displacements = peaks_displacement[:, 0]

#         # For each unique reference peak, average the displacement values
#         # to prevent inconsistencies in interpolation
#         unique_peaks = np.unique(ref_peaks)
#         averaged_displacements = np.array([
#             np.mean(displacements[ref_peaks == peak]) for peak in unique_peaks
#         ])

#         peak_displacement_mapping = np.vstack([unique_peaks, averaged_displacements]).T
#         for peak, displacement in peak_displacement_mapping:
#             mask = ref_peaks == peak
#             peaks_displacement[mask, 0] = displacement

#         return peaks_displacement, peak_displacement_mapping
    
#     peaks_displacement, hap = update_displacements(peaks_ref, peaks_displacement)

#     # Linear interpolation inside the convex hull
#     hum = interp1d(hap[:, 0], hap[:, 1], kind="linear", fill_value="extrapolate")(
#         np.arange(target.shape[1])
#     )

#     # Use linear extrapolation outside the convex hull to ensure continuity.
#     pks, displ1d = peaks_ref[:, 0], peaks_displacement[:, 0]
#     min_pks, max_pks = np.min(pks), np.max(pks)
#     hum2 = interp1d(
#         [min_pks, max_pks],
#         [
#             np.mean(displ1d[pks == min_pks]),
#             np.mean(displ1d[pks == max_pks]),
#         ],
#         kind="linear",
#         fill_value="extrapolate",
#     )(np.arange(-1, target.shape[1] + 2))

#     # Populate the displacement matrix
#     min_x_pksref, max_x_pksref = np.min(peaks_ref[:, 0]), np.max(peaks_ref[:, 0])
#     for w in range(aligned.shape[0]):
#         for x in range(aligned.shape[1]):
#             if aligned[w, x] != 0:
#                 continue
#             if model_choice == "normal":
#                 displacement[w, x, :] = [
#                     fdist1[int(w - wo[0]), int(x * peak_ratio - xo[0])],
#                     hum[x] if min_x_pksref <= x <= max_x_pksref else hum2[x + 1],
#                 ]
#             else:
#                 displacement[w, x, :] = [
#                     fdist1[int(w - wo[0]), int(x * peak_ratio - xo[0])],
#                     fdist2[int(w - wo[0]), int(x * peak_ratio - xo[0])],
#                 ]
#     del fdist1, fdist2

#     def apply_interpolation(ref, target, displacement, method="linear"):
#         """
#         Applies interpolation to realign the chromatogram.
#         The function constructs an extended target grid to allow for interpolation near edges.
#         It applies the computed displacement to get new pixel locations.
#         Finally, it interpolates the new values using interpn to realign the chromatogram.
        
#         Parameters:
#             ref (numpy.ndarray): Reference chromatogram.
#             target (numpy.ndarray): Target chromatogram.
#             displacement (numpy.ndarray): Displacement matrix.
#             method (str): Interpolation method.
        
#         Returns:
#             numpy.ndarray: Aligned chromatogram.
#         """
#         # Prepare grid for interpolation
#         rh, rw = ref.shape
#         th, tw = target.shape

#         X = np.tile(np.arange(rw), (rh * 2, 1))
#         h_half = round(rh / 2)
#         Y = np.arange(-h_half, rh + rh - h_half).reshape(-1, 1)
#         Y = np.tile(Y, (1, rw))

#         Z = np.zeros((th * 2, tw))
#         mid_floor, mid_ceil = int(np.floor(th / 2)), int(np.ceil(th / 2))

#         Z[:mid_floor, 1:] = target[mid_ceil:, :-1]
#         Z[mid_floor : mid_floor + th, :] = target
#         Z[mid_floor + th :, :-1] = target[:mid_ceil, 1:]

#         mid_idx = round(th / 2)
#         Xq = X[mid_idx : mid_idx + th, :] + displacement[:, :, 1]
#         Yq = Y[mid_idx : mid_idx + th, :] + displacement[:, :, 0]

#         # 2d interpolation
#         points = (X[0, :], Y[:, 0])
#         queries = np.column_stack((Xq.ravel(), Yq.ravel()))

#         method = "splinef2d" if method == "spline" else method
#         aligned = interpn(
#             points, Z.T, queries, method=method, fill_value=0, bounds_error=False
#         )
#         aligned = aligned.reshape(target.shape)
#         return aligned

#     aligned = apply_interpolation(
#         ref, target, displacement, method=inter_pixel_interp_meth
#     )

#     # -- Extension with Borders: Add padding and interpolate displacement at image boundaries.
#     displacement[peaks_ref[:, 1], peaks_ref[:, 0], :] = peaks_displacement[:, ::-1]
#     displacement_extended = np.pad(
#         displacement, pad_width=((1, 1), (1, 1), (0, 0)), mode="constant"
#     )

#     natw, natx = peaks_ref_corner[:, 1] + 1, (peaks_ref_corner[:, 0] + 1) * peak_ratio
#     natv = peaks_displacement_corner[:, 1]
#     wo = np.arange(-padding_w_lower, aligned.shape[0] + 2 + padding_w_upper)
#     xo = np.arange(
#         np.round(-padding_x_lower * peak_ratio),
#         np.round((aligned.shape[1] + 2 + padding_x_upper) * peak_ratio),
#     )
#     fdist1bis = natgrid(natw, natx, natv, wo, xo)

#     for w in (0, aligned.shape[0] + 1):
#         for x in range(aligned.shape[1] + 2):
#             displacement_extended[w, x, :] = [
#                 fdist1bis[int(w - wo[0]), int(x * peak_ratio - xo[0])],
#                 hum[x] if min_x_pksref <= x <= max_x_pksref else hum2[x],
#             ]

#     for w in range(1, aligned.shape[0] + 1):
#         for x in (0, aligned.shape[1] + 1):
#             displacement_extended[w, x, :] = [
#                 fdist1bis[int(w - wo[0]), int(x * peak_ratio - xo[0])],
#                 hum[x] if min_x_pksref <= x <= max_x_pksref else hum2[x],
#             ]
#     del fdist1bis, hum, hum2
#     gc.collect()

#     # -- Compute deformation correction
#     #   -> Preserves relative pixel intensities: Adjusts for compression/expansion in displacement.
#     #   Compression: Pixels are pushed together, leading to intensity overestimation due to overlap.
#     #   Expansion: Pixels are spread apart, causing intensity underestimation as the image is "diluted"
#     #   over a larger area.

#     # Initialize deformation arrays
#     deform1 = np.zeros_like(aligned)
#     deform2 = np.zeros_like(aligned)

#     # Compute deformation
#     for w in range(aligned.shape[0]):
#         for x in range(aligned.shape[1]):
#             deform1[w, x] = 2 + (
#                 -displacement_extended[w, x + 1, 0]
#                 + displacement_extended[w + 2, x + 1, 0]
#             )
#             deform2[w, x] = 2 + (
#                 -displacement_extended[w + 1, x, 1]
#                 + displacement_extended[w + 1, x + 2, 1]
#             )

#     # Correct for negative deformations
#     if np.any(deform1 < 0) or np.any(deform2 < 0):
#         print(
#             "Warning: Some displacement predictions lead to inversion of pixel order (as indicated by negative values in Deform_output, "
#             "here set to zero) please change your set of alignment points!"
#         )
#         deform1[deform1 < 0] = 0
#         deform2[deform2 < 0] = 0

#     # Apply deformation correction
#     aligned *= (deform1 * deform2) / 4

#     deform_output = np.stack((deform1, deform2), axis=-1)

#     return aligned, displacement, deform_output


# def equalize_size(chromato1, chromato2):
#     """
#     Equalizes the size of two GCxGC chromatograms by adding zeros where needed.

#     Parameters:
#         chromato1 (numpy.ndarray): First chromatogram.
#         chromato2 (numpy.ndarray): Second chromatogram.

#     Returns:
#         tuple: Two chromatograms of equal size (numpy.ndarrays).
#     """
#     if chromato1.shape != chromato2.shape:
#         # Determine the maximum size for the output chromatograms
#         max_shape = np.maximum(chromato1.shape, chromato2.shape)

#         # Initialize new chromatograms with zeros
#         chromato1_eq = np.zeros(max_shape)
#         chromato2_eq = np.zeros(max_shape)

#         # Fill in the original data
#         chromato1_eq[: chromato1.shape[0], : chromato1.shape[1]] = chromato1
#         chromato2_eq[: chromato2.shape[0], : chromato2.shape[1]] = chromato2
#     else:
#         chromato1_eq = chromato1
#         chromato2_eq = chromato2

#     return chromato1_eq, chromato2_eq


# def save_chromatogram(filename, chromato_obj, replace_existing=False):
#     """
#     Save the chromatogram data to a NetCDF file.

#     Parameters:
#         filename (str): Path to the output NetCDF file.
#         chromato_obj (dict): Dictionary containing the chromatogram data.
#     """
#     # Get the directory path from the filename
#     directory = os.path.dirname(filename)

#     # Create the directory if it doesn't exist
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)

#     # Add a timestamp to the filename if it already exists
#     if not replace_existing and os.path.exists(filename):
#         filename, file_extension = os.path.splitext(filename)
#         timestamp = time.strftime("_%Y%m%d_%H%M%S")
#         filename = filename + timestamp + file_extension

#     # Create a new NetCDF file with 64-bit offset
#     with Dataset(filename, "w", format="NETCDF4") as ncnew:
#         # Define dimensions
#         scan_number = ncnew.createDimension(
#             "scan_number", len(chromato_obj["scantime"])
#         )
#         point_number = ncnew.createDimension(
#             "point_number", len(chromato_obj["MSintbox"])
#         )

#         # Define variables
#         vardimfirst = ncnew.createVariable("flag_count", "i4", (scan_number,))
#         vardim_a = ncnew.createVariable("scan_acquisition_time", "f8", (scan_number,))
#         vardim_b = ncnew.createVariable("actual_scan_number", "i4", (scan_number,))
#         vardim_c = ncnew.createVariable("mass_range_max", "f8", (scan_number,))
#         vardim_d = ncnew.createVariable("mass_range_min", "f8", (scan_number,))
#         vardim_e = ncnew.createVariable("scan_index", "i4", (scan_number,))
#         vardim_f = ncnew.createVariable("point_count", "i4", (scan_number,))
#         vardim_g = ncnew.createVariable("total_intensity", "f8", (scan_number,))
#         vardim_h = ncnew.createVariable("mass_values", "f8", (point_number,))
#         vardim_i = ncnew.createVariable("intensity_values", "f8", (point_number,))

#         # Write data to variables
#         vardimfirst[:] = chromato_obj["flag"]
#         vardim_a[:] = chromato_obj["scantime"]
#         vardim_b[:] = chromato_obj["scannum"]
#         vardim_c[:] = chromato_obj["medmsmax"]
#         vardim_d[:] = chromato_obj["medmsmin"]
#         vardim_e[:] = chromato_obj["ionid"]
#         vardim_f[:] = chromato_obj["eachscannum"]
#         vardim_g[:] = chromato_obj["MStotint"]
#         vardim_h[:] = chromato_obj["MSvaluebox"]
#         vardim_i[:] = chromato_obj["MSintbox"]


# def run_chromatogram_alignment(config):
#     """
#     Main function to run the chromatogram alignment process.

#     Parameters:
#         config (dict): Dictionary containing the configuration parameters.
#     """
#     print("Running the main script.")
#     print("Loading chromatograms and alignment points...")
#     chromato_target, chromato_ref = load_chromatograms(
#         input_path=config["io_params"]["INPUT_PATH"],
#         target_file=config["io_params"]["TARGET_CHROMATOGRAM_FILE"],
#         reference_file=config["io_params"]["REFERENCE_CHROMATOGRAM_FILE"],
#         int_thresh=config["instrument_params"]["INTTHRESHOLD"],
#         drift_ms=config["instrument_params"]["DRIFTMS"],
#     )
#     print("Chromatograms loaded successfully.")

#     print("Loading alignment points...")
#     reference_peaks, target_peaks = load_alignment_points(
#         chromato_ref=chromato_ref, chromato_target=chromato_target, config=config
#     )
#     print("Alignment points loaded successfully.")

#     # Compute the number of pixels in the 2nd dimension based on the modulation time and sampling rate
#     nb_pix_2nd_d_ref = int(
#         config["instrument_params"]["MODTIME"] * chromato_ref["SamRate"]
#     )
#     nb_pix_2nd_d_target = int(
#         config["instrument_params"]["MODTIME"] * chromato_target["SamRate"]
#     )

#     print("Reshaping chromatogram data...")
#     ref_tic = reshape_tic(chromato_ref["MStotint"], nb_pix_2nd_d_ref)
#     target_tic = reshape_tic(chromato_target["MStotint"], nb_pix_2nd_d_target)
#     print("Chromatogram data reshaped successfully.")
#     del chromato_ref
#     gc.collect()

#     print("Rounding MS data...")
#     time_start = time.time()
#     chromato_target["MSvaluebox"], chromato_target["MSintbox"] = (
#         round_and_aggregate_unique_ms_data(
#             chromato_target["MSvaluebox"],
#             chromato_target["MSintbox"],
#             precision=config["instrument_params"]["PRECISION"],
#         )
#     )
#     time_end = time.time()
#     print("MS data rounded successfully. Time taken:", time_end - time_start)

#     aligned_result = align_2d_chrom_ms_v5(
#         ref=ref_tic,
#         other=target_tic,
#         peaks_ref=reference_peaks,
#         peaks_other=target_peaks,
#         ms_valuebox=chromato_target["MSvaluebox"],
#         ms_intbox=chromato_target["MSintbox"],
#         nb_pix_2nd_d=nb_pix_2nd_d_ref,
#         peak_widths=config["model_choice_params"]["TYPICAL_PEAK_WIDTH"],
#         model_choice=config["model_choice_params"]["MODEL_CHOICE"],
#     )
#     del ref_tic, target_tic
#     gc.collect()

#     aligned_each_scan_num = np.sum(aligned_result["MSvaluebox"] != 0, axis=1)
#     aligned_ion_id = np.cumsum(aligned_each_scan_num)
#     aligned_ms_tot_int = np.sum(aligned_result["MSintbox"], axis=1)

#     aligned_result["scannum"] = chromato_target["scannum"]
#     aligned_result["flag"] = chromato_target["flag"]
#     aligned_result["medmsmax"] = chromato_target["medmsmax"]
#     aligned_result["medmsmin"] = chromato_target["medmsmin"]
#     aligned_result["scantime"] = chromato_target["scantime"]
#     aligned_result["eachscannum"] = aligned_each_scan_num
#     aligned_result["ionid"] = aligned_ion_id
#     aligned_result["MStotint"] = aligned_ms_tot_int

#     # Flatten the MSvaluebox and MSintbox arrays and remove zeros
#     ms_valuebox = aligned_result["MSvaluebox"].ravel()
#     ms_intbox = aligned_result["MSintbox"].ravel()
#     mask = ms_valuebox != 0

#     aligned_result["MSvaluebox"] = ms_valuebox[mask]
#     aligned_result["MSintbox"] = ms_intbox[mask]

#     print("Saving aligned chromatogram...")
#     output_file_name = os.path.join(
#         config["io_params"]["OUTPUT_PATH"],
#         os.path.splitext(config["io_params"]["TARGET_CHROMATOGRAM_FILE"])[0]
#         + "_ALIGNED.cdf",
#     )
#     save_chromatogram(output_file_name, aligned_result)
#     print("Aligned chromatogram saved successfully.")


# def combine_swpa_pbp_alignment(config, num_points=10, align_all_targets=False):
#     """
#     Combine the SWPA and PBP alignment methods.

#     Parameters:
#         config (dict): Dictionary containing the configuration parameters.
#         num_points (int): Number of alignment points to use.
#         align_all_targets (bool): Whether to align all target chromatograms.
#     """
#     input_dir = config["io_params"]["INPUT_PATH"]
#     ref_filename = config["io_params"]["REFERENCE_CHROMATOGRAM_FILE"]

#     # -- Run SWPA peak alignment
#     with ProcessPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(
#             swpa_peak_alignment,
#             input_dir,
#             ref_filename,
#             mod_time=config["instrument_params"]["MODTIME"],
#         )
#         matches = future.result()

#     # -- Run PBP alignment
#     if align_all_targets:
#         target_files = [
#             f for f in os.listdir(input_dir) if f.endswith(".cdf") and f != ref_filename
#         ]
#     else:
#         target_files = [config["io_params"]["TARGET_CHROMATOGRAM_FILE"]]

#     def coordinates_to_csv(matches, num_points, output_dir, config):
#         """
#         Save the alignment points to a CSV file.

#         Parameters:
#             matches (pd.DataFrame): DataFrame containing the alignment points.
#             num_points (int): Number of alignment points to save.
#             output_dir (str): Path to the output directory.
#             config (dict): Dictionary containing the configuration parameters.
#         """
#         output_file_ref = os.path.join(
#             output_dir,
#             config["io_params"]["REFERENCE_ALIGNMENT_PTS_FILE"],
#         )
#         output_file_target = os.path.join(
#             output_dir,
#             config["io_params"]["TARGET_ALIGNMENT_PTS_FILE"],
#         )
#         coordinates = matches[
#             matches["filename"]
#             == os.path.splitext(config["io_params"]["TARGET_CHROMATOGRAM_FILE"])[0]
#         ]

#         # Ensure num_points does not exceed the number of available rows
#         num_points = min(num_points, len(coordinates))

#         coordinates = coordinates.head(num_points)

#         units = config["model_choice_params"]["UNITS"]
#         if units == "pixel":
#             ref_coord = coordinates[["rp1", "rp2"]]
#             target_coord = coordinates[["tp1", "tp2"]]
#         else:
#             ref_coord = coordinates[["rt1", "rt2"]]
#             target_coord = coordinates[["tt1", "tt2"]]

#         ref_coord.to_csv(output_file_ref, index=False, header=False)
#         target_coord.to_csv(output_file_target, index=False, header=False)

#     for target_file in target_files:
#         config["io_params"]["TARGET_CHROMATOGRAM_FILE"] = target_file
#         coordinates_to_csv(
#             matches, num_points=num_points, output_dir=input_dir, config=config
#         )
#         run_chromatogram_alignment(config)


# if __name__ == "__main__":
#     config_path = os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         "..",
#         "config",
#         "pbp_pixel_config.json",
#     )
#     config = load_config(config_path)
#     run_chromatogram_alignment(config)
