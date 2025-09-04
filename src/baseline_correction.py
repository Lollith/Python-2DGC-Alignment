import numpy as np
import multiprocessing
from scipy.signal import savgol_filter
import pybaselines
from whittaker_eilers import WhittakerSmoother
from scipy.ndimage import map_coordinates
from pybaselines import Baseline2D


def method_poly(chromato, order=3, lamb=0.5):
    baseline_fitter = Baseline2D()
    bs, params = baseline_fitter.poly(chromato,poly_order=order)
    correct=chromato-bs
    correct=smooth2d((correct), lmbd=lamb, lmbd2=lamb)
    if(np.all(correct==0)) :
        return correct
    correct[correct<=0]=min(correct[correct>0])
    return correct


def smooth2d(arr, lmbd=25,lmbd2=25):
    # Lissage par lignes
    smoother = WhittakerSmoother(lmbda=lmbd2, order=1, data_length=arr.shape[1])
    arr_smooth = np.array([smoother.smooth(row) for row in arr])

    # Lissage par colonnes
    smoother = WhittakerSmoother(lmbda=lmbd, order=1, data_length=arr.shape[0])
    arr_smooth = np.array([smoother.smooth(col) for col in arr_smooth.T]).T
    
    return arr_smooth

def method_window(mat,block_size = 20, gamma=0.5, lmbd=25):
    
    if(np.all(mat==0)) :
        return mat
    chrom=mat
    
    mat = smooth2d(mat, lmbd=lmbd,lmbd2=lmbd)
    n=mat.shape[0]
    m=mat.shape[1]
    n_blocks = (n + block_size - 1) // block_size
    m_blocks = (m + block_size - 1) // block_size

    min_grid = np.zeros((n_blocks, m_blocks))

    for bi, i in enumerate(range(0, n, block_size)):
        for bj, j in enumerate(range(0, m, block_size)):
            block = mat[i:i+block_size, j:j+block_size]
            min_grid[bi, bj] = block.min()

    min_grid_smooth = smooth2d(min_grid, lmbd=25)
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
    values = np.array(values) + (3*gamma-1)*sigma

    # Grille complète (n x m)
    grid_x, grid_y = np.mgrid[0:n, 0:m]

    # --- 4. Interpolation linéaire 2D ---
    interp_mat = map_coordinates(min_grid_smooth, [grid_x/block_size, grid_y/block_size], order=1, mode='nearest')

    #interp_mat = griddata(points, values, (grid_x, grid_y), method='nearest')

    correct=smooth2d(chrom, lmbd=0.5, lmbd2=0.5)
    correct=correct-interp_mat
    correct[correct<0]=min(correct[correct>0])
    
    return(correct)

from scipy.signal import savgol_filter
import pybaselines

def method_als(chromato, sg_windows=5,lam=10**3,p=0.01):
    
    chromato= savgol_filter(chromato,
           window_length=sg_windows,  # 5, 11 pour un lissage + fort
           polyorder=3,
           mode='nearest')
    
    bs_mat = np.empty_like(chromato)
    for i in range(bs_mat.shape[1]):
        bs=pybaselines.whittaker.asls(chromato[:, i],lam=lam,p=p)[0]
        bs_mat[:, i] =  bs
                                
    bs_mat=savgol_filter(bs_mat,window_length=sg_windows,polyorder=2,
           mode='nearest')                                                
    tmp=chromato-bs_mat
    if(np.all(tmp==0)) :
        return tmp
    tmp[tmp<=0]=min(tmp[tmp>0])
    return tmp

def baseline_correct(chromato,method):
    if method =="als":
        correct = method_als(chromato,  sg_windows=10,lam=10**3,p=0.001)
    if method == "window":
        correct = method_window(chromato,block_size=10,gamma=0.5, lmbd=25)
    if method == "poly":
        correct = method_poly(chromato)
    return correct

def chromato_cube_corrected_baseline(chromato_cube,method):
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
        for i, result in enumerate(pool.starmap(baseline_correct, [(m_chromato, method) for m_chromato in (chromato_cube)])):
            chromato_cube_no_baseline.append(result)
    return chromato_cube_no_baseline