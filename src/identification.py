import read_chroma
# import mass_spec
# import baseline_correction
import peak_detection
import matching
import integration
import csv
import numpy as np
# from skimage.restoration import estimate_sigma
# import os
import traceback
# import argparse


def write_line(compound_name, rt1, rt2, area, formatted_spectrum):
    return compound_name + "\t" + "\"" + str(rt1 * 60) + " , " + str(rt2) + "\"" + "\t" + str(area) + "\t" + "T" + "\t" + formatted_spectrum + "\n"

# def takeSecond(elem):
#     return elem[1]


def mass_spectra_format(mass_range, int_values):
    """
    Formats the mass spectrum data into a string representation, sorting the
    mass-to-intensity pairs by intensity in descending order. Each pair is
    represented as "mass:intensity" and separated by spaces.

    Parameters:
    -----------
    mass_range : tuple of int
        A tuple representing the range of mass values, in the form (min_mass,
        max_mass),
        where the mass values should be within this range.

    int_values : ndarray
        A numpy array containing the intensity values corresponding to the
        mass values in `mass_range`.

    Returns:
    --------
    str
        A formatted string where each mass-to-intensity pair is represented as
        "mass:intensity", with pairs sorted in descending order of intensity.

    Example:
    --------
    >>> mass_range = (100, 110)
    >>> int_values = np.array([10, 20, 15, 25, 30, 5, 40, 10, 12, 8])
    >>> result = mass_spectra_format(mass_range, int_values)
    >>> print(result)
    "110:40 109:30 108:25 107:20 106:15 105:12 104:10 103:10 102:8 101:5"
    """

    range_min, range_max = mass_range
    mass_values = np.linspace(
        range_min, range_max, range_max - range_min + 1).astype(int)
    spectrum = np.column_stack((mass_values, int_values))
    sorted_by_int_spectrum = spectrum[(-spectrum[:, 1]).argsort()]
    formatted_spectrum = ""
    for i, mz_int in enumerate(sorted_by_int_spectrum):
        if (i != 0):
            formatted_spectrum = formatted_spectrum + " "
        formatted_spectrum = (formatted_spectrum + str(mz_int[0]) + ":" +
                              str(mz_int[1]))
    return formatted_spectrum


def compute_matches_identification(matches, chromato, chromato_cube,
                                   mass_range, formated_spectra,
                                   similarity_threshold=0.001,
                                   ):
    """
    Computes the identification data for each match in a list of matches,
    including the integration of peak areas and heights, along with additional
    compound information. Optionally formats the spectra associated with each
    match.

    Parameters:
    -----------
    matches : list of tuples
        A list of matches, where each match is a tuple containing:
        - (RT1, RT2): The retention times of the match.
        - A dictionary containing compound data such as 'casno',
        'compound_name', 'compound_formula',
          'hit_prob', 'match_factor', 'reverse_match_factor', and 'spectra'.
        - A coordinate tuple (coord) representing the position of the match.

    chromato : ndarray
        A 2D array representing the chromatogram data with intensity values
        for each point.

    chromato_cube : ndarray
        A 3D array representing the chromatogram cube, containing spectral
        data for each chromatogram point.

    mass_range : tuple of int
        A tuple representing the range of mass values (min_mass, max_mass) for
        the spectrum formatting.

    similarity_threshold : float, optional, default=0.001
        The threshold for similarity when checking peak pool similarity. A
        lower value will result in stricter matching criteria.

    formated_spectra : bool, optional, default=False
        If True, formats and includes the mass spectra in the identification
        data for each match.

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing the identification data for a
        match, including:
        - 'casno', 'compound_name', 'compound_formula', 'hit_prob',
        'match_factor', 'reverse_match_factor', 'rt1', 'rt2', 'area', 'height',
        and optionally 'spectra' (if `formated_spectra` is True).

    Example:
    --------
    >>> matches = [( (5.2, 5.3), {'casno': '123-45-6', 'compound_name':
        'Compound A', 'compound_formula': 'C6H12O6',
        'hit_prob': 0.95, 'match_factor': 0.98, 'reverse_match_factor': 0.97,
        'spectra': [100, 200, 150]}, (3, 4))]
    >>> chromato = np.array([[0, 1], [2, 3]])
    >>> chromato_cube = np.random.rand(10, 2, 2)  # Example 3D data
    >>> mass_range = (50, 150)
    >>> result = compute_matches_identification(matches, chromato,
        chromato_cube, mass_range)
    >>> print(result)
    """

    matches_identification = []

    #TODO : ajout ici
    # Trouver la longueur maximale des éléments dans matches
    max_len = max(len(match) for match in matches)

    # Compléter les lignes plus courtes ou tronquer les lignes trop longues
    matches = [match + [None] * (max_len - len(match)) if len(match) < max_len else match[:max_len] for match in matches]
    matches = np.array(matches, dtype=object) #TODO

    for match in matches:
        coord = match[2]
        blob = integration.peak_pool_similarity_check(
            chromato, np.stack(matches[:, 2]), coord, chromato_cube,
            threshold=0.5, plot_labels=True,
            similarity_threshold=similarity_threshold)
        area = integration.compute_area(chromato, blob)
        height = chromato[coord[0], coord[1]]

        identification_data_dict = dict()
        identification_data_dict['casno'] = match[1]['casno']
        identification_data_dict['compound_name'] = match[1]['compound_name']
        identification_data_dict['compound_formula'] = \
            (match[1]['compound_formula'])
        identification_data_dict['hit_prob'] = match[1]['hit_prob']
        identification_data_dict['match_factor'] = match[1]['match_factor']
        identification_data_dict['reverse_match_factor'] = \
            (match[1]['reverse_match_factor'])
        identification_data_dict['rt1'] = match[0][0]
        identification_data_dict['rt2'] = match[0][1]
        identification_data_dict['area'] = area
        identification_data_dict['height'] = height
        if (formated_spectra):
            identification_data_dict['spectra'] = \
                (mass_spectra_format(mass_range, match[1]['spectra']))
        matches_identification.append(identification_data_dict)

    return matches_identification


def identification(filename, mod_time, method, mode, seuil, hit_prob_min,
                   ABS_THRESHOLDS, cluster, min_distance, sigma_ratio,
                   num_sigma, formated_spectra, match_factor_min):
    r"""Takes a chromatogram as file and returns identified compounds.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    method : optional
        Method used to detect peaks.
    mode : optional
        Mode used to detect peaks. Can be either tic or mass_per_mass or 3D.
    seuil : optional
        Used to compute theshold as seuil * estimated gaussian white noise.
    hit_prob_min : optional
        Filter compounds with hit_prob < hit_prob_min
    ABS_THRESHOLDS : optional
        If mode='mass_per_mass' or mode='3D', ABS_THRESHOLDS is the threshold
        relative to a slice of the 3D chromatogram or a slice of the 3D
        chromatogram.
    cluster : optional
        Whether to cluster coordinates when mode is mass_per_mass or 3D.
    min_distance : optional
        peak_local_max method parameter. The minimal allowed distance
        separating peaks. To find the maximum number of peaks, use
        min_distance=1.
    sigma_ratio : optional
        DoG method parameter. The ratio between the standard deviation of
        Gaussian Kernels used for computing the Difference of Gaussians.
    num_sigma : optional
        LoG/DoH method parameter. The number of intermediate values of
        standard deviations to consider between min_sigma (1) and max_sigma
        (30).
    formated_spectra : optional
        If spectra need to be formatted for peak table based alignment.
    match_factor_min : optional
        Filter compounds with match_factor < match_factor_min.
    -------
    Returns
    -------
    matches_identification:
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    --------
    """
    chromato, time_rn, chromato_cube, sigma, mass_range = (
        read_chroma.read_chromato_and_chromato_cube(filename, mod_time,
                                                    pre_process=True
                                                    ))
    MIN_SEUIL = seuil * sigma * 100 / np.max(chromato)
    # find 2D peaks
    coordinates = peak_detection.peak_detection(
        (chromato, time_rn, mass_range), None, chromato_cube, MIN_SEUIL,
        ABS_THRESHOLDS, method=method, mode=mode, cluster=cluster,
        min_distance=min_distance, sigma_ratio=sigma_ratio,
        num_sigma=num_sigma)
    print("nb peaks", len(coordinates))

    # 2D peaks identification
    matches = matching.matching_nist_lib_from_chromato_cube(
        (chromato, time_rn, mass_range), chromato_cube, coordinates,
        mod_time, hit_prob_min=hit_prob_min,
        match_factor_min=match_factor_min)
    print("nb match", len(matches))

    matches_identification = compute_matches_identification(
        matches, chromato, chromato_cube, mass_range,
        formated_spectra, similarity_threshold=0.001)
    return matches_identification


def cohort_identification_to_csv(filename, matches_identification, PATH):
    r"""Generate csv (readable) peak table.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    matches_identification :
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    PATH : optional
        Path to the resulting formatted peak table.

    Returns
    -------
    None
        The function writes a CSV file, containing one line per identified 
        compound:
        - Name : Chemical compound name (e.g., Toluene)
        - Casno : CAS number (unique compound identifier)
        - Formula : Molecular formula (e.g., C7H8)
        - hit_prob : Hit probability (%), confidence in the identification
        - match_factor : Match factor between observed and library spectra
        - reverse_match_factor : Reverse match factor ignoring unmatched peaks
        in the sample
        - rt1 : Retention time in the 1st dimension
        - rt2 : Retention time in the 2nd dimension
        - Area : Peak area (proportional to the compound abundance)
        - Height : Peak height (related to concentration)
    """

    with open(PATH + filename + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # header
        writer.writerow(['Name', 'Casno', 'Formula', 'hit_prob',
                         'match_factor', 'reverse_match_factor', 'rt1', 'rt2',
                         'Area', 'Height'])

        for identification_data_dict in matches_identification:
            casno = identification_data_dict['casno']
            compound_name = identification_data_dict['compound_name']
            compound_formula = identification_data_dict['compound_formula']
            hit_prob = identification_data_dict['hit_prob']
            match_factor = identification_data_dict['match_factor']
            reverse_match_factor = \
                (identification_data_dict['reverse_match_factor'])
            rt1 = identification_data_dict['rt1']
            rt2 = identification_data_dict['rt2']
            area = identification_data_dict['area']
            height = identification_data_dict['height']
            row = [compound_name, casno, compound_formula, hit_prob,
                   match_factor, reverse_match_factor, rt1, rt2, area, height]
            writer.writerow(row)


def cohort_identification_alignment_input_format_txt(
        filename, matches_identification, PATH):
    r"""Generate formatted peak table for alignment.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    matches_identification :
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    PATH : optional
        Path to the resulting formatted peak table.
    """
    with open(PATH + filename + '.txt', 'w', encoding='UTF8') as f:
        f.write("Name\tR.T...s.\tArea\tQuant.Masses\tSpectra\n")
        for identification_data_dict in matches_identification:
            compound_name = identification_data_dict['compound_name']
            rt1 = identification_data_dict['rt1']
            rt2 = identification_data_dict['rt2']
            area = identification_data_dict['area']
            formatted_spectrum = identification_data_dict['spectra']
            f.write(write_line(compound_name, rt1, rt2, area,
                               formatted_spectrum))


def sample_identification(path, file, OUTPUT_PATH, mod_time, method, mode,
                          seuil, hit_prob_min, ABS_THRESHOLDS, cluster,
                          min_distance, sigma_ratio, num_sigma,
                          formated_spectra, match_factor_min):
    r"""Read sample chromatogram and generate the associated peak table.
    - identification()

    Parameters
    ----------
    path : str
        Path to the directory containing the chromatogram file.
    file : str
        Name of the chromatogram file.
    OUTPUT_PATH : str, optional
        Directory where the resulting peak table files will be saved. If None,
        results are saved in the current working directory.
    mod_time : float, default=1.25
        Modulation time used for chromatogram processing.
    method : str, default='persistent_homology'
        Method used for peak detection.
    mode : str, default='tic'
        Mode of chromatogram analysis.
    seuil : int, default=5
        Detection threshold for peaks.
    hit_prob_min : int, default=15
        Minimum hit probability for compound identification.
    ABS_THRESHOLDS : list or None, default=None
        Absolute intensity thresholds for peak detection.
    cluster : bool, default=True
        Whether to apply clustering to detected peaks.
    min_distance : int, default=1
        Minimum distance between detected peaks.
    sigma_ratio : float, default=1.6
        Ratio used for Gaussian peak fitting.
    num_sigma : int, default=10
        Number of standard deviations to consider for peak detection.
    formated_spectra : bool, default=True
        Whether to format spectra before identification.
    match_factor_min : int, default=700
        Minimum match factor for compound identification.

    Examples
    --------
    >>> sample_identification("/path/to/data/", "sample.cdf")
    >>> # or with an output directory
    >>> sample_identification("/path/to/data/", "sample.cdf",
        OUTPUT_PATH="/path/to/results/")
    """

    print('Identification started\n')
    try:
        full_filename = path + file
        matches_identification = \
            identification(full_filename,
                           mod_time=mod_time,
                           method=method,
                           mode=mode,
                           seuil=seuil,
                           hit_prob_min=hit_prob_min,
                           ABS_THRESHOLDS=ABS_THRESHOLDS,
                           cluster=cluster,
                           min_distance=min_distance,
                           sigma_ratio=sigma_ratio,
                           num_sigma=num_sigma,
                           formated_spectra=formated_spectra,
                           match_factor_min=match_factor_min)
        print("Identification done")
        if (OUTPUT_PATH is not None):
            cohort_identification_alignment_input_format_txt(
                file[:-4], matches_identification, OUTPUT_PATH)
            print('.txt created')
            cohort_identification_to_csv(file[:-4], matches_identification,
                                         OUTPUT_PATH)
            print(".csv created")
        else:
            cohort_identification_alignment_input_format_txt(
                file[:-4], matches_identification)
            print('.txt created')
            cohort_identification_to_csv(file[:-4], matches_identification)
            print(".csv created")

    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file}: {e}")
        traceback.print_exc()


# def read_process_input_and_launch_sample_identification(PATH, filename, OUTPUT_PATH, method='persistent_homology', mode='tic', seuil=5, hit_prob_min=15, ABS_THRESHOLDS=None, cluster=True, min_distance=1, sigma_ratio=1.6, num_sigma=10, formated_spectra=True, match_factor_min=700):

#     r"""Read chromatogram and generate peak table. Executed in subprocess to avoid memory leaks due to NIST search.

#     Parameters
#     ----------
#     PATH :
#         Path to the chromatogram.
#     file :
#         Filename of the chromatogram.
#     OUTPUT_PATH : optional
#         Path to the resulting peak table.
#     --------
#     Examples
#     --------
#     >>> python identification.py 'PATH' 'file'
#     >>> # or
#     >>> python identification.py 'PATH' 'file' 'OUTPUT_PATH'
#     """
#     '''if (len(sys.argv) > 2):
#         PATH=sys.argv[1]
#         file=sys.argv[2]
#         if (len(sys.argv) > 3):
#             OUTPUT_PATH=sys.argv[3]
#             sample_identification(PATH, file, OUTPUT_PATH)
#         else:
#             print(PATH, file)
#             sample_identification(PATH, file)'''
            
#     sample_identification(PATH, filename, OUTPUT_PATH, method=method, mode=mode, seuil=seuil, hit_prob_min=hit_prob_min, ABS_THRESHOLDS=ABS_THRESHOLDS, cluster=cluster, min_distance=min_distance, sigma_ratio=sigma_ratio, num_sigma=num_sigma, formated_spectra=formated_spectra, match_factor_min=match_factor_min)

    
# def cohort_identification(path):
#     files = os.listdir(path)
#     name_files_list = []
    
#     for file in files:

#         try:
#             print(file)
#             name_files_list.append(file[:-4])
#             full_filename = path + file
#             matches_identification = identification(full_filename, hit_prob_min=35, formated_spectra=True)
#             print("identification done")
#             cohort_identification_alignment_input_format_txt(file[:-4], matches_identification)
#             print("csv created")
#         except:
#             print('error', file)
#             traceback.print_exc()

#         try:
#             del matches_identification
#         except:
#             continue

# if __name__ == '__main__':

#     # read parameters
#     parser=argparse.ArgumentParser(description="Launch sample identification")
#     # required parameters
#     parser.add_argument('-p', '--path', required=True, help="Path to the directory containing the chromatograms of the cohort")
#     parser.add_argument('-f', '--filename', required=True, help="Filename of the cdf chromatogram")
#     parser.add_argument('-op', '--output_path', required=True, help="Path where peaks table will be generated. The input path for the alignment and path where aligned peak table will be generated")

#     # optionals parameters
#     parser.add_argument('-m', '--method', default='persistent_homology', help="Method used to detect peaks")
#     parser.add_argument('--mod_time', default=1.25, type=float, help="Modulation time")
#     parser.add_argument('--mode', default='tic', help="Mode used to detect peaks. Can be either tic or mass_per_mass or 3D.")
#     parser.add_argument('--match_factor_min', type=int, default=0, help="Filter compounds with match_factor < match_factor_min")
#     parser.add_argument('-t', '--threshold', type=float, default=5, help="Used to compute theshold as threshold * 100 * estimated gaussian white noise / np.max(chromato).")
#     parser.add_argument('-hpm', '--hit_prob_min', type=int, default=0, help="Filter compounds with hit_prob < hit_prob_min")
#     parser.add_argument('-at', '--abs_threshold', default=None, type=float, help="If mode='mass_per_mass' or mode='3D', ABS_THRESHOLDS is the threshold relative to a slice of the 3D chromatogram or a slice of the 3D chromatogram.")
#     parser.add_argument('-c', '--cluster', default=True, type=bool, help="Whether to cluster coordinates when mode is mass_per_mass or 3D.")
#     parser.add_argument('-md', '--min_distance', default=1, type=int, help="peak_local_max method parameter. The minimal allowed distance separating peaks. To find the maximum number of peaks, use min_distance=1.")
#     parser.add_argument('-sr', '--sigma_ratio', default=1.6, type=float, help="DoG method parameter. The ratio between the standard deviation of Gaussian Kernels used for computing the Difference of Gaussians.")
#     parser.add_argument('-ns', '--num_sigma', default=10, type=int, help="LoG/DoH method parameter. The number of intermediate values of standard deviations to consider between min_sigma (1) and max_sigma (30).")
#     parser.add_argument('-fs', '--format_spectra', default=True, type=bool, help="If spectra need to be formatted for peak table based alignment.")
#     args=parser.parse_args()


#     print(args.path, type(args.path))
#     print(args.filename, type(args.path))
#     print(args.output_path, type(args.output_path))
#     print(args.mod_time, type(args.mod_time))
#     print(args.method, type(args.method))
#     print(args.mode, type(args.mode))
#     print(args.match_factor_min, type(args.match_factor_min))
#     print(args.threshold, type(args.threshold))
#     print(args.hit_prob_min, type(args.hit_prob_min))
#     print(args.abs_threshold, type(args.abs_threshold))
#     print(args.cluster, type(args.cluster))
#     print(args.min_distance, type(args.min_distance))
#     print(args.sigma_ratio, type(args.sigma_ratio))
#     print(args.num_sigma, type(args.num_sigma))
#     print(args.format_spectra, type(args.format_spectra))

#     print('start identification')

#     #read_process_input_and_launch_sample_identification(args.path, args.filename, args.output_path, method=args.method, mode=args.mode, seuil=args.threshold, hit_prob_min=args.hit_prob_min, ABS_THRESHOLDS=args.abs_threshold, cluster=args.cluster, min_distance=args.min_distance, sigma_ratio=args.sigma_ratio, num_sigma=args.num_sigma, formated_spectra=args.format_spectra, match_factor_min=args.match_factor_min)
#     sample_identification(args.path, args.filename, args.output_path, mod_time=args.mod_time, method=args.method, mode=args.mode, seuil=args.threshold, hit_prob_min=args.hit_prob_min, ABS_THRESHOLDS=args.abs_threshold, cluster=args.cluster, min_distance=args.min_distance, sigma_ratio=args.sigma_ratio, num_sigma=args.num_sigma, formated_spectra=args.format_spectra, match_factor_min=args.match_factor_min)
