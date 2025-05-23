# Python-2DGC Functions Overview

<style>
.bg_color {
    background-color: aliceblue;
    border-radius: 3px;
    font-size: 85%;
    line-height: 1.45;
    overflow: auto;
    padding: 16px;
}
</style>

# Help on module identification:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>identification(filename, method='persistent_homology', mode='tic', seuil=5, hit_prob_min=15, ABS_THRESHOLDS=None, cluster=True, min_distance=1, sigma_ratio=1.6, num_sigma=10, formated_spectra=True, match_factor_min=700)</div>
        Takes a chromatogram as file and returns identified compounds.
</code></pre>

    cohort_identification(path)
    
    cohort_identification_alignment_input_format_txt(filename, matches_identification, PATH='./COVID-2020-2021/')
        Generate formatted peak table for alignment.

    cohort_identification_to_csv(filename, matches_identification, PATH='./COVID-2020-2021/')
        Generate csv (readable) peak table.
        
    compute_matches_identification(matches, chromato, chromato_cube, mass_range, similarity_threshold=0.001, formated_spectra=False)
    
    mass_spectra_format(mass_range, int_values)
    
    read_process_input_and_launch_sample_identification()
        Read chromatogram and generate peak table. Executed in subprocess to avoid memory leaks due to NIST search.
        
    
    sample_identification(path, file, OUTPUT_PATH=None)
        Read sample chromatogram and generate associated peak table.
        
    takeSecond(elem)
    
    write_line(compound_name, rt1, rt2, area, formatted_spectrum)

# Help on module read_chroma:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>read_chroma(filename, mod_time=1.25, max_val=None)</div>
        Read chromatogram file.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>read_chromato_and_chromato_cube(filename, mod_time=1.25, pre_process=True)</div>
        Same as read_chromato_cube but do not returns full spectra_obj (only range_min and range_max) because of RAM issue.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>read_chromato_cube(filename, mod_time=1.25, pre_process=True)</div>
        Read chromatogram file and compute TIC chromatogram, 3D chromatogram and noise std.
</code></pre>

    full_spectra_to_chromato_cube(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None)
        Compute 3D chromatogram from mass spectra. Then it is possible to read specific mass spectrum from this 3D chromatogram or detect peaks in 3D.

    centroided_to_mass_nominal_chromatogram(filename, cdf_name, mod_time=1.25)
        Read centroided chromatogram, convert and save it as nominal mass chromatogramm cdf file.
    
    full_spectra_to_chromato_cube_centroid(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None)
    
    full_spectra_to_chromato_cube_from_centroid(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None)
        Compute 3D chromatogram from centroided mass spectra to handle centroided details.

    print_chroma_header(filename)
        Print chromato header.

    read_chroma_spectra_loc(filename, mod_time=1.25)
        Read chromatogram file.

    read_only_chroma(filename, mod_time=1.25)
        Read chromatogram file.

    chromato_cube(spectra_obj, mass_range_min=None, mass_range_max=None, debuts=None)
    
    chromato_part(chromato_obj, rt1, rt2, mod_time=1.25, rt1_window=5, rt2_window=0.1)

    read_chromato_mass(spectra_obj, tm, debuts=None)

# Help on module plot:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>visualizer(chromato_obj, mod_time=1.25, rt1=None, rt2=None, rt1_window=5, rt2_window=0.1, plotly=False, title='', points=None, radius=None, pt_shape='.', log_chromato=True, casnos_dict=None, contour=[], center_pt=None, center_pt_window_1=None, center_pt_window_2=None, save=False)</div>
        Plot chromatogram
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>plot_mass(mass_values, int_values, title='', top_n_mass=10, figsize=(32, 18))</div>
        Plot mass spectrum
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>plot_3d_chromato(chromato, rstride=10, cstride=10, plot_map=True)</div></code></pre>

    mass_overlay(mass_values_list, intensity_values_list, title='mass_overlay', top_n_mass=10, figsize=(32, 18))
        Plot multiple mass spectra

    get_cmap(n, name='hsv')
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.

    plot_acp(features_disc_mol_new_cd, labels, projection=None, figsize=(10, 5))
        Plot ACP
        
    plot_confusion_matrix(conf_mat)
    
    plot_corr_matrix(corr_matrix)
    
    plot_feature_and_permutation_importance(feature_importance, permutation_importance, mol_list, id_max=10)
    
    plot_feature_corr(index, corr_matrix, mol_list, id_max=20)
        Plot the correlation of a feature with all other feature

    plot_feature_importance(feature_importance, mol_list, id_max=10)
    
    plot_hmdb_id_spectra(path, hmdb_id)

    plot_scores_array(scores_array, similarity_measure)
    
    point_is_visible(point, indexes)


# Help on module mass_spec:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>read_full_spectra_centroid(spectra_obj, max_val=None)</div>
        Build nominal mass mass spectra from centroided mass and intensity values.
</code></pre>

    centroid(spectra_obj, mv_index, mass_values, int_values)
    
    centroid_to_full_nominal(spectra_obj, mass_values, int_values)
    
    compute_chromato_mass_from_spectra_loc(loc, mass_range, mass)
    
    filter_spectra(file)
    
    hmdb_txt_to_mgf(path, libname='./hmdb_lib.mgf')
    
    peak_retrieval(spectrums, spectrums_lib, coordinates_in_chromato, min_score=None)
    
    peak_retrieval_from_path(spectrum_path, lib_path, coordinates_in_chromato=None)
    
    peak_retrieval_kernel(spectra, spectrums_lib, similarity_measure)
    
    peak_retrieval_multiprocessing(spectrums, spectrums_lib, coordinates_in_chromato, min_score=None)
    
    read_full_spectra(spectra_obj, max_val=None)
    
    read_full_spectra_full_centroid(spectra_obj, max_val=None)
    
    read_hmdb_spectrum(filename)
    
    read_spectra(spectra_obj, max_val=None)
    
    read_spectra_centroid(spectra_obj, max_val=None)
    
    read_spectrum(chromato, pic_coord, spectra)
    
    read_spectrum_from_chromato_cube(pic_coord, chromato_cube)
    
    read_spectrum_tr_coord(chromato, pic_coord, spectra, time_rn, mod_time=1.25)
    
    spectra_matching(spectrums, spectrums_lib, similarity_measure=<matchms.similarity.CosineGreedy.CosineGreedy object at 0x0000016F61063AF0>)


# Help on module baseline_correction:

**_FUNCTIONS_**


<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>chromato_cube_corrected_baseline(chromato_cube)</div>
    Apply baseline correction on each chromato of the input.

</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>chromato_no_baseline(chromato, j=None)</div>
        Correct baseline and apply savgol filter.
</code></pre>
    
    baseline_als(y, lam, p, niter=10)


# Help on module discriminant_pixel:

**_FUNCTIONS_**
    
    
<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>find_discriminant_compounds(chromato_ref_obj, aligned_chromatos, chromato_cube, vij, disp=False, max_pixel=500, local_max_filter=False, mod_time=1.25, title='', hit_prob_min=15, match_factor_min=800)</div>
        Finds discriminant compounds (finds discriminant pixels and identifies them).
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>find_discriminant_pixels(chromato_ref_obj, aligned_chromatos, vij, disp=False, max_pixel=500, local_max_filter=False, title='')</div>
        Find discriminant pixels.
</code></pre>

    read_aligned_chromato(filename)
    
    read_aligned_chromatos(CORRECTED_CHROMA_PATH)

    compute_map(chromatos, vij)

# Help on module integration:

**_FUNCTIONS_**


<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>peak_pool_similarity_check(chromato, coordinates, coordinate, chromato_cube, threshold=0.25, similarity_threshold=0.01, plot_labels=False)</div>
        Peak integration. Find pixels in the neighborhood of the peak passed in parameter which belong to the peak blob. It is usefull to determine the area of the peak.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>peak_pool_similarity_check_coordinates(chromato, coordinates, coordinate, chromato_cube, threshold=0.25, similarity_threshold=0.01, plot_labels=False)</div>
        Peak integration. Find pixels in the neighborhood of the peak passed in parameter which belong to the peak blob. It is usefull to determine the area of the peak.
</code></pre>

    compute_area(chromato, blob)
        Compute area of a blob.
    
    estimate_num_components(X, max_components=3)
    
    estimate_num_components_aic(X, max_components=3)
    
    estimate_num_components_with_spectra_dist(X, chromato_cube, mass_values, max_components=3, threshold=0.9)
    
    estimate_peaks_center_aic(cds, num_components=None, max_components=3)
    
    estimate_peaks_center_bic(cds, num_components=None, max_components=3)
    
    estimate_peaks_center_with_dist(cds, chromato_cube, mass_values, num_components=None, threshold=0.9, max_components=3)
    
    get_all_area(chromato, coordinates, threshold=0.25)
    
    get_all_contour(blobs)
    
    get_contour(blob, chromato, time_rn)
    
    peak_pool(chromato, coordinates, coordinate, threshold=0.25, plot_labels=False)
    
    similarity_cluestering(chromato_cube, coordinates, ref_point, similarity_threshold=0.01)
    
    tmp(chromato, coordinates, coordinate, threshold=0.25)

# Help on module matching:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>matching_nist_lib_from_chromato_cube(chromato_obj, chromato_cube, coordinates, mod_time=1.25, hit_prob_min=15, match_factor_min=800)</div>
        Indentify retrieved peaks using NIST library.
</code></pre>

    matching_nist_lib(chromato_obj, spectra, coordinates, mod_time=1.25)

    check_match(match)
    
    compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25)
    
    compute_metrics_from_chromato_cube(coordinates, chromato_obj, chromato_cube, spectrum_path, spectrum_lib, mod_time=1.25)
    
    matching(chromato_obj, spectra, spectrum_path, spectrum_lib, coordinates, mod_time=1.25, min_score=None)
    

# Help on module peak_simulation:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>simulation_from_cdf_model(lib_path='./lib_EIB.mgf', scores_path='./lib_scores.json', min_similarity=0.0, max_similarity=1.1, min_overlap=0.7, max_overlap=0.99, intensity_range_min=20000, intensity_range_max=40000, model_filename='./data/ELO_CDF/model.cdf', nb_chromato=1, mod_time=1.25, new_cdf_path='./data/ELO_CDF/', cdf_name='new_cdf', noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9)</div>
        Creates a new chromatogram with overlapped clusters based on a model chromatogram and saves it as cdf file.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>create_chromato_cube_with_overlapped_cluster(shape, lib_spectra, array_scores, min_overlap=0.7, max_overlap=0.99, min_similarity=0.0, max_similarity=1.1, add_noise=True, intensity_range_min=20000, intensity_range_max=40000, noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9)</div>
        Create a 3D chromatogram with overlapped clusters
</code></pre>

    add_noise(noise_typ, image)
        Print chromato header.
    
    add_noise_3D(image, noise_typ='custom_gauss', noise_loc=1000.0, noise_scale=500.0, poisson_rep=0.9)
        Add noise in 3D chromatogram
    
    add_peak(chromato, peak, mu)
        Take a chromatogram and a peak and add the peak at the location mu in the TIC chromatogram.
        
    add_peak_and_spectrum(chromato, chromato_cube, peak, mu)
    
    asym_gauss_kernel(size: int, sizey: int = None, sigma=1.0) -> <built-in function array>
        Returns a 2D asymetric Gaussian kernel.
    
    create_and_add_peak(chromato, size, intensity, mu, sigma=1.0)
        Take a chromatogram, create a peak and add the peak at the location mu in the TIC chromatogram.

    create_overlapped_cluster_in_cube(mu0, min_overlap=0.7, max_overlap=0.99)
        Create overlapped cluster.
    
    create_peak(size, intensity, sigma=1.0, random_noise=False)
        Returns a peak (guassien kernel multiply by intensity).
    
    create_random_blob(mu_x, mu_y, mu_x_range=0, mu_y_range=0, sigma_x_range_min=0.5, sigma_x_range_max=5, sigma_y_range_min=0.5, sigma_y_range_max=5)
        Create random blob.

    add_chromato_cube_gaussian_white_noise(chromato_cube, noise_typ='gauss')
    
    create_and_add_peak_spectrum(chromato_cube, size, intensity, mu, spectrum, sigma=1.0)
    
    create_chromato_cube(chromato_shape, nb_peaks, mu_center=[], center_dist=3, size=80, min_similarity=0.0, max_similarity=1.1, lib_path='./lib_EIB.mgf', scores_path='./lib_scores.json')
    
    create_chromato_with_overlapped_cluster(shape, min_overlap=0.7, max_overlap=0.99, sigma_x_range_min=2, sigma_x_range_max=5, sigma_y_range_min=2, sigma_y_range_max=5)
    
    create_overlapped_cluster(chrom, mu0, sigma_x_range_min=2, sigma_x_range_max=5, sigma_y_range_min=2, sigma_y_range_max=5, size=40, min_overlap=0.7, max_overlap=0.99)
    
    create_random_mass_spectra(nb)
    
    delete_peak(chromato_cube, size, intensity, mu, spectrum, sigma=1.0)
    
    gauss_kernel(size: int, sizey: int = None, sigma=1.0) -> <built-in function array>
        Returns a 2D Gaussian kernel.

    get_similar_spectrum(spectrum_index, array_scores, min_similarity=0.0, max_similariy=1.1)


# Help on module peak_table_alignment:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>peak_table_alignment()</div>
        Peak table based alignment.
</code></pre>

# Help on module pixel_alignment:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>pixel_alignment(PATH='F:/Bureau/032023-data-Tenax-VOC-COVID/')</div>
        Pixel based chromatogram alignment.
</code></pre>

# Help on module pbp_alignment:

**_CONFIG_**
```json
{
  "instrument_parameters": {
    "PRECISION": 2,  // Decimal precision for m/z values (e.g., 2 for 0.01 precision)
    "INT_THRESHOLD": 0,  // Signal intensity threshold (integer or float)
    "MODTIME": 8.1, // Modulation time in seconds (float)
    "DRIFT_MS": 0  // Drift mass spectrum adjustment (integer). If it is set to "1", all the m/z values will be added "1" (e.g., m/z 300 => m/z 301)
  },
  "model_choice_parameters": {
    "TYPICAL_PEAK_WIDTH": [1, 5],  // Typical peak widths in pixels or time units
    "MODEL_CHOICE": "normal",  // Alignment model: "normal" or "DualSibson"
    "UNITS": "pixel"  // Units for peak width: "pixel" or "time"
  },
  "io_parameters": {
    "INPUT_PATH": "/path/to/input/",  // Directory for input chromatograms
    "OUTPUT_PATH": "/path/to/output/",  // Directory for output files
    "REFERENCE_CHROMATOGRAM_FILE": "reference.cdf",  // Reference chromatogram filename
    "TARGET_CHROMATOGRAM_FILE": "target.cdf",  // Target chromatogram filename
    "REFERENCE_ALIGNMENT_PTS_FILE": "alignment_ref.csv",  // Reference alignment points
    "TARGET_ALIGNMENT_PTS_FILE": "alignment_target.csv"  // Target alignment points
  }
}
```

# Help on module utils:

**_FUNCTIONS_**

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>compute_spectra_similarity(spectrum1, spectrum2)</div>
        Compute similarity between two spectra. First spectrum and second spectrum must have the same dimensions.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>get_two_points_and_compute_spectra_similarity(pt1, pt2, spectra_obj, chromato_cube)</div>
        Get two points and compute spectra similarity.
</code></pre>

<pre class="bg_color"><code><div style="color: red; font-weight: bold;";>retrieve_formula_and_mass_from_compound_name(compound_name)</div></code></pre>

    add_formula_in_aligned_peak_table(filename, output_filename)
        Retrieve and add formula in aligned peak table for each molecule. 

    add_formula_weight_in_aligned_peak_table_from_molecule_name_formula_dict(filename, output_filename, molecule_name_formula_dict)
        Add formula weight in aligned peak table from molecule name/formula dict

    add_nist_info_in_mgf_file(filename, output_filename='lib_EIB_gt.mgf')
        Retrieve NIST infos of spectra in HMDB library file (MGF format) and create a new library (MGF format) with retrieved NIST informations.

    build_molecule_name_formula_dict_from_peak_tables(PATH)
        Take aligned peak table and create a dictionary which associate formula and nominal mass to each molecule in the peak table
    
    compute_spectrum_mass_values(spectra_obj)
        Compute nominal mass format mass values from spectra infos. NEED REFACTO

    formula_to_nominal_mass(formula)
        Retrieve and add formula in aligned peak table for each molecule.

    retrieved_nist_casnos_from_hmbd_spectra(lib_path)
        NIST Search with HMDB spectra to retrieve NIST infos (casnos...).

    shuffle_dict(d)
        Shuffle dictionary

    get_name_dict(matches)
        Group coordinates in matches by compound name

    colors(coordinates, matches)

    get_casno_dict(matches)
    
    get_casno_list(matches)
    
    get_name_list(matches)
        
    retrieve_mol_list_formula_weight(mol_list, mol_list_formla_weight_dict)

    unique_mol_list_formla_weight_dict(mol_list)
    
# Help on module projection:

**_FUNCTIONS_**

    chromato_to_matrix(points, time_rn, mod_time, chromato_dim)
        Project points from chromatogram (in time) into matrix chromatogram (ndarray).

    matrix_to_chromato(points, time_rn, mod_time, chromato_dim)
        Project points from chromatogram matrix (ndarray) into chromatogram (in time).


# Help on module peaks_shape:

**_FUNCTIONS_**

    calculate_moment(img, width, height, p, q) -> int
        Renvoie le moment d'ordre (p + q) de l'image f de taille (width * height).
    
    central_moment(img, width, height, p, q) -> int
        Renvoie le moment central d'ordre (p + q) de l'image f de taille (width * height) centrÚ sur le centre de gravitÚ de l'image.
        Le moment central est insensible aux translations.
    
    centre_of_gravity(img, width, height)
        Renvoie les coordonnÚes du centre de gravitÚ (x, y) de la forme de l'image f de taille (width * height).
    
    rotation_moment(img, width, height)
        Renvoie les 7 moments insensibles aux rotations et Ó l'Úchelle de l'image f de taille (width * height) centrÚ sur le centre de gravitÚ de l'image.
    
    scale_moment(img, width, height, mu_00)
        Renvoie une sÚlÚction de moments insensibles aux translations d'ordres prÚdÚfinis de l'image f de taille (width * height) centrÚ sur le centre de gravitÚ de l'image.

# Help on module write_masspec:

**_FUNCTIONS_**

    MSP_files_to_MGF_file(path, filename_mgf=None)
        Create Mascot format file from multiple MSP files.

    mass_spectra_to_mgf(filename, mass_values_list, intensity_values_list, meta_list=None, filter_by_instrument_type=None, names=None, casnos=None, hmdb_ids=None)
        Write mass spectra in Mascot format file.

    nist_msl_to_mgf(filename, new_filename=None)
        Convert NIST MSL file to Mascot file format.
