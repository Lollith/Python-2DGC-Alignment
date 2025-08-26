import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import multiprocessing
from scipy.signal import savgol_filter
import pybaselines

import time
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata
from whittaker_eilers import WhittakerSmoother


def smooth2d(arr, lmbd=25):
    # Lissage par lignes
    smoother = WhittakerSmoother(lmbda=lmbd, order=1, data_length=arr.shape[1])
    arr_smooth = np.array([smoother.smooth(row) for row in arr])

    # Lissage par colonnes
    smoother = WhittakerSmoother(lmbda=lmbd, order=1, data_length=arr.shape[0])
    arr_smooth = np.array([smoother.smooth(col) for col in arr_smooth.T]).T
    
    return arr_smooth

def baseline_correct_window(mat,j=None,block_size = 20,gamma=0.25,lmbd=25):
    
    mat= savgol_filter(mat,
           window_length=10,  # 5, 11 pour un lissage + fort
           polyorder=3,
           mode='nearest')
    
    n=mat.shape[0]
    m=mat.shape[1]
    n_blocks = (n + block_size - 1) // block_size
    m_blocks = (m + block_size - 1) // block_size

    min_grid = np.zeros((n_blocks, m_blocks))

    for bi, i in enumerate(range(0, n, block_size)):
        for bj, j in enumerate(range(0, m, block_size)):
            block = mat[i:i+block_size, j:j+block_size]
            min_grid[bi, bj] = block.min()

    min_grid_smooth = smooth2d(min_grid, lmbd=lmbd)
    sigma=np.std(min_grid)

    # --- 3. Préparer pour interpolation ---
    points = []
    values = []
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            ci = bi*block_size + block_size//2
            cj = bj*block_size + block_size//2
            if ci < n and cj < m:  # éviter dépassement
                points.append((ci, cj))
                values.append(min_grid_smooth[bi, bj])

    points = np.array(points)
    values = np.array(values) + gamma*sigma

    # Grille complète (n x m)
    grid_x, grid_y = np.mgrid[0:n, 0:m]

    # --- 4. Interpolation linéaire 2D ---
    interp_mat = griddata(points, values, (grid_x, grid_y), method='nearest')


    correct=mat-interp_mat
    correct[correct<0]=min(correct[correct>0])
    return(correct)


def baseline_correct_window_cube(chromato_cube):
    r"""Apply baseline correction on each chromato of the input.
    ----------
    chromato_cube :
        Input chromato.
    Returns
    -------
    chromato_cube:
        List of chromato from input list without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
    """
    #cpu_count = multiprocessing.cpu_count()
    cpu_count = min(multiprocessing.cpu_count(), 32) #TODO
    chromato_cube_no_baseline = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for i, result in enumerate(pool.starmap(baseline_correct_window, [(m_chromato, j) for j, m_chromato in enumerate(chromato_cube)])):
            chromato_cube_no_baseline.append(result)
    return chromato_cube_no_baseline

# def baseline_als(y, lam, p, niter=10):

#   L = len(y)
#   D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
#   w = np.ones(L)
#   for i in range(niter):
#     W = sparse.spdiags(w, 0, L, L)
#     Z = W + lam * D.dot(D.transpose())
#     z = spsolve(Z, w*y)
#     w = p * (y > z) + (1-p) * (y < z)
#   return z


# def chromato_no_baseline(chromato, j=None): #rename
def chromato_reduced_noise(chromato, j=None,sg_windows=5):
    r"""Correct baseline and apply savgol filter.
    ----------
    chromato : ndarray
        Input chromato.
    Returns
    -------
    chromato :
        The input chromato without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> chromato = baseline_correction.chromato_no_baseline(chromato)
    """

    chromato= savgol_filter(chromato,
           window_length=sg_windows,  # 5, 11 pour un lissage + fort
           polyorder=3,
           mode='nearest')
    
    tmp = np.empty_like(chromato)
    for i in range(tmp.shape[1]):
        tmp[:, i] = chromato[:, i] - pybaselines.whittaker.asls(chromato[:, i],
                                                       lam=10**3,
                                                       p=0.01)[0]
    tmp[tmp < .0] = 0

    tmp[tmp<0]=0
    return tmp


def chromato_cube_corrected_baseline(chromato_cube,sg_windows=50):
    r"""Apply baseline correction on each chromato of the input.
    ----------
    chromato_cube :
        Input chromato.
    Returns
    -------
    chromato_cube:
        List of chromato from input list without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
    """
    #cpu_count = multiprocessing.cpu_count()
    cpu_count = min(multiprocessing.cpu_count(), 32) #TODO
    chromato_cube_no_baseline = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for i, result in enumerate(pool.starmap(chromato_reduced_noise, [(m_chromato, j,sg_windows) for j, m_chromato in enumerate(chromato_cube)])):
            chromato_cube_no_baseline.append(result)
    return chromato_cube_no_baseline