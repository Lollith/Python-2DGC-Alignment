import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import projection 
import multiprocessing
import numpy as np
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

def baseline_correct(mat,block_size = 20,gamma=0.25,lmbd=15):
    
    # mat= savgol_filter(mat,
    #        window_length=100,  # 5, 11 pour un lissage + fort
    #        polyorder=3,
    #        mode='nearest')
    
    if(np.all(mat==0)) :
        return mat
    
    mat = smooth2d(mat, lmbd=lmbd)

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
    values = np.array(values) + gamma*sigma

    # Grille complète (n x m)
    grid_x, grid_y = np.mgrid[0:n, 0:m]

    # --- 4. Interpolation linéaire 2D ---
    interp_mat = griddata(points, values, (grid_x, grid_y), method='nearest')


    correct=mat-interp_mat
    correct[correct<0]=min(correct[correct>0])
    return(correct)


# --------- 2. Estimation des sigmas gauche/droite/haut/bas à partir du FWHM ---------
def estimate_sigma_from_FWHM_asym(image, center):
    """
    Calcule sigma_x1 (gauche), sigma_x2 (droite), sigma_y1 (bas), sigma_y2 (haut)
    à partir du FWHM sur chaque demi-axe.
    """
    cy, cx = center
    profile_x = image[cy, :]
    profile_y = image[:, cx]

    half_max_x = profile_x[cx] / 2.0
    half_max_y = profile_y[cy] / 2.0

    # --- X gauche
    left_idx = np.where(profile_x[:cx] < half_max_x)[0]
    x_left = left_idx[-1] if len(left_idx) > 0 else 0
    fwhm_left = (cx - x_left) * 2
    sigma_x1 = fwhm_left / 2.355 if fwhm_left > 0 else 1.0

    # --- X droite
    right_idx = np.where(profile_x[cx:] < half_max_x)[0]
    x_right = right_idx[0] + cx if len(right_idx) > 0 else len(profile_x)-1
    fwhm_right = (x_right - cx) * 2
    sigma_x2 = fwhm_right / 2.355 if fwhm_right > 0 else 1.0

    # --- Y bas
    down_idx = np.where(profile_y[:cy] < half_max_y)[0]
    y_down = down_idx[-1] if len(down_idx) > 0 else 0
    fwhm_down = (cy - y_down) * 2
    sigma_y1 = fwhm_down / 2.355 if fwhm_down > 0 else 1.0

    # --- Y haut
    up_idx = np.where(profile_y[cy:] < half_max_y)[0]
    y_up = up_idx[0] + cy if len(up_idx) > 0 else len(profile_y)-1
    fwhm_up = (y_up - cy) * 2
    sigma_y2 = fwhm_up / 2.355 if fwhm_up > 0 else 1.0

    return sigma_x1, sigma_x2, sigma_y1, sigma_y2

# --------- 3. Multi-gaussienne pour lmfit ---------

def twoD_Gaussian_bi_asym_fixed_center(coords, amp, sx1, sx2, sy1, sy2, xo, yo):
    x, y = coords
    dx = x - xo
    dy = y - yo
    sigma_x = np.full_like(dx, sx2, dtype=float)
    sigma_x[dx < 0] = sx1
    sigma_y = np.full_like(dy, sy2, dtype=float)
    sigma_y[dy < 0] = sy1
    return amp * np.exp(-0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2))

def multi_gaussian_lmfit(x, y, **params):
    z = np.zeros_like(x, dtype=float)
    n_peaks = len(fixed_centers)
    for i, (yo, xo) in enumerate(fixed_centers):  # (y, x)
        amp = params[f'amp_{i}']
        sx1 = params[f'sx1_{i}']
        sx2 = params[f'sx2_{i}']
        sy1 = params[f'sy1_{i}']
        sy2 = params[f'sy2_{i}']
        z += twoD_Gaussian_bi_asym_fixed_center((x, y), amp, sx1, sx2, sy1, sy2, xo, yo).reshape(x.shape)
    return z.ravel()

def fit_all_peaks_fixed_centers(tmp, coords, bounds, plot=False):
    ny, nx = tmp.shape
    fitted_results = []

    # --- Fenêtre de travail (comme avant) ---
    min_x = bounds[1][0]
    max_x = bounds[1][1]
    min_y = max(bounds[0][0], 0)
    max_y = min(ny, bounds[0][1])

    if (min_x < 0):
        if min_y > 0:
            sub_tmp = np.hstack([tmp[(min_y-1):(max_y-1), (nx+min_x):nx], tmp[min_y:max_y, 0:max_x]])
        else :
            sub_tmp = tmp[min_y:max_y, 0:max_x]
    elif (max_x > nx) :
        if  max_y < (ny):
            sub_tmp = np.hstack([tmp[min_y:max_y, min_x:nx], tmp[(min_y+1):(max_y+1), 0:(max_x-nx)]])
        else: sub_tmp = tmp[min_y:max_y, min_x:nx]
    else:
        sub_tmp = tmp[min_y:max_y, min_x:max_x]

    H, W = sub_tmp.shape

    # --- Grille LOCALE (0..W-1, 0..H-1) -> centres et (x,y) dans le même repère ---
    xx_loc, yy_loc = np.meshgrid(np.arange(W), np.arange(H))

    # --- lmfit setup ---
    global fixed_centers
    fixed_centers = []
    params = Parameters()

    for i, (yy0_g, xx0_g) in enumerate(coords):
        # map global -> local
        # x : modulo pour tolérer un éventuel "wrap" horizontal ; y : clamp dans la fenêtre
        xx0 = (xx0_g - min_x) % W
        yy0 = int(np.clip(yy0_g - min_y, 0, H-1))

        amp0 = float(sub_tmp[yy0, xx0])

        # Estimation des sigmas dans la SOUS-FENÊTRE (indépendant de tmp global)
        sx1, sx2, sy1, sy2 = estimate_sigma_from_FWHM_asym(sub_tmp, (yy0, xx0))

        # Bornes un peu plus larges pour stabiliser le fit (évite un plafonnement trop tôt)
        local_max = float(sub_tmp.max())
        params.add(f'amp_{i}', value=amp0, min=0.0, max=max(amp0*3.0, local_max*1.2) + 1e-6)
        params.add(f'sx1_{i}',  value=sx1, min=0, max=max(sx1*2, 1e-6))
        params.add(f'sx2_{i}',  value=sx2, min=0, max=max(sx2*2, 1e-6))
        params.add(f'sy1_{i}',  value=sy1, min=0, max=max(sy1*2, 1e-6))
        params.add(f'sy2_{i}',  value=sy2, min=0, max=max(sy2*2, 1e-6))

        fixed_centers.append((yy0, xx0))  # centres en repère local

    # --- Fit sur la SOUS-FENÊTRE en repère local ---
    model = Model(multi_gaussian_lmfit, independent_vars=['x', 'y'])
    result = model.fit(
        sub_tmp.ravel(),
        x=xx_loc,
        y=yy_loc,
        params=params,
        method='leastsq',
        max_nfev=500  # un peu plus d'itérations
    )

    # --- Reconstructions & métriques ---
    fitted_peak_list = []
    for i, (yy0, xx0) in enumerate(fixed_centers):
        amp = result.params[f'amp_{i}'].value
        sx1 = result.params[f'sx1_{i}'].value
        sx2 = result.params[f'sx2_{i}'].value
        sy1 = result.params[f'sy1_{i}'].value
        sy2 = result.params[f'sy2_{i}'].value

        peak = twoD_Gaussian_bi_asym_fixed_center(
            (xx_loc, yy_loc), amp, sx1, sx2, sy1, sy2, xx0, yy0
        ).reshape(sub_tmp.shape)
        fitted_peak_list.append(peak)

        # renvoi des centres en coordonnées globales (utile pour la suite du pipeline)
        yy_g = yy0 + min_y
        xx_g = (xx0 + min_x) % nx

        fitted_results.append({
            "group_id": i,
            "center": (yy_g, xx_g),
            "params": (amp, sx1, sx2, sy1, sy2),
            "area": float(np.sum(peak))
        })

    fitted_data = np.sum(fitted_peak_list, axis=0) if fitted_peak_list else np.zeros_like(sub_tmp)

    y_true = sub_tmp.ravel().astype(float)
    y_pred = fitted_data.ravel().astype(float)
    sst = np.sum((y_true - y_true.mean())**2)
    ssr = np.sum((y_true - y_pred)**2)
    r2_total = 1.0 - ssr / sst if sst > 0 else float('nan')

    for rec in fitted_results:
        rec["r2_total"] = float(r2_total)

    if plot:
        vmin, vmax = np.min(sub_tmp), np.max(sub_tmp)
        fig, axes = plt.subplots(1, 2 + len(fitted_peak_list), figsize=(3 * (2 + len(fitted_peak_list)), 4))

        axes[0].contourf(sub_tmp.T, vmin=vmin, vmax=vmax)
        for (cy, cx) in fixed_centers:
            axes[0].plot(cy, cx, "ro", markersize=5)
        axes[0].set_title("Original (fenêtre locale)")

        axes[1].contourf(fitted_data.T, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Fit total (R² = {r2_total:.4f})")

        for k, peak in enumerate(fitted_peak_list):
            axes[2 + k].contourf(peak.T, vmin=vmin, vmax=vmax)
            axes[2 + k].set_title(f"Pic {k + 1}")

        plt.tight_layout()
        plt.show()

    return fitted_results


def compute_bounds(coords, radii):
    """
    Calcule les bornes [min,max] en RT et m/z pour chaque pic
    coords: array (n,2) -> (rt, mz)
    radii: array (n,) -> rayon du blob
    """
    bounds = []
    for (rt, mz), radius in zip(coords, radii):

        rt_radius = radius * 2 * 1.7 / 60
        mz_radius = radius * 6 * 0.005

        rt_min, rt_max = rt - rt_radius, rt + rt_radius
        mz_min, mz_max = mz - mz_radius, mz + mz_radius

        bounds.append([[rt_min, rt_max], [mz_min, mz_max]])

    return np.array(bounds)  # shape (n, 2, 2)


def is_inside(coord, bound):
    """Vérifie si un point (rt, mz) est à l'intérieur d'un bound"""
    rt, mz = coord
    inside_rt = bound[0][0] <= rt <= bound[0][1]
    inside_mz = bound[1][0] <= mz <= bound[1][1]
    return inside_rt and inside_mz


def cluster_bounds(coords, bounds, time_rn, mod_time, chromato_shape, max_size=4):
    """
    Regroupe les pics si le centre d'un pic tombe dans le bound d'un autre,
    puis subdivise les clusters trop grands avec DBSCAN jusqu'à max_size.
    """
    n = len(bounds)
    visited = [False] * n
    clusters = []

    # --- Étape 1 : clustering initial par inclusion des bounds ---
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        stack = [i]
        visited[i] = True

        while stack:
            j = stack.pop()
            for k in range(n):
                if not visited[k]:
                    if is_inside(coords[k], bounds[j]) or is_inside(coords[j], bounds[k]):
                        visited[k] = True
                        stack.append(k)
                        cluster.append(k)

        clusters.append(cluster)

    # --- Préparer les coordonnées normalisées ---
    rt_vals = projection.matrix_to_chromato(coords, time_rn, mod_time, chromato_shape)
    rt_vals[:, 0] = rt_vals[:, 0] * 60
    scaled_rt = rt_vals / np.array([10, 0.15])

    # --- Étape 2 : subdivision des clusters trop grands ---
    final_clusters = []

    for cluster in clusters:
        if len(cluster) <= max_size:
            final_clusters.append(cluster)
        else:
            stack = [cluster]

            while stack:
                current_cluster = stack.pop()

                if len(current_cluster) <= max_size:
                    final_clusters.append(current_cluster)
                else:
                    sub_coords = scaled_rt[current_cluster]
                    sub_penalty_matrix = 0.01 * cdist(sub_coords, sub_coords, metric='cityblock')

                    # DBSCAN adaptatif
                    eps = 0.005
                    sub_labels = None
                    for _ in range(5):  # max 5 tentatives
                        sub_clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(sub_penalty_matrix)
                        sub_labels = sub_clustering.labels_
                        if len(set(sub_labels)) > 1:
                            break
                        eps /= 10  # réduire eps si pas de subdivision

                    # Si DBSCAN échoue (1 seul cluster), on force la sortie
                    if len(set(sub_labels)) == 1:
                        final_clusters.append(current_cluster)
                        continue

                    # Ajouter les sous-clusters
                    for sub_label in set(sub_labels):
                        sub_idx = np.where(sub_labels == sub_label)[0]
                        sub_cluster = [current_cluster[i] for i in sub_idx]
                        if len(sub_cluster) > max_size:
                            stack.append(sub_cluster)
                        else:
                            final_clusters.append(sub_cluster)

    return final_clusters


def expand_cluster_bounds(bounds, clusters):
    """Élargit les bornes de chaque cluster"""
    cluster_bounds = []
    for cluster in clusters:
        rt_min = min(bounds[i][0][0] for i in cluster)
        rt_max = max(bounds[i][0][1] for i in cluster)
        mz_min = min(bounds[i][1][0] for i in cluster)
        mz_max = max(bounds[i][1][1] for i in cluster)
        cluster_bounds.append([[rt_min, rt_max], [mz_min, mz_max]])
    return np.array(cluster_bounds)


def construct_spectrum(quanti_all_mass, label, nmass):
    spec_list = []
    area_list = []

    for clust in label:
        spec = np.zeros(nmass)
        max_heights = {}

        for j in clust:
            mass = int(quanti_all_mass[j][0])  
            height = quanti_all_mass[j][2]

            # Garde la hauteur maximale pour chaque masse
            if mass not in max_heights or height > max_heights[mass]:
                max_heights[mass] = height

        # Remplir le spectre avec les hauteurs maximales
        for mass, height in max_heights.items():
            spec[mass] = height

        # Trouver l’aire du pic le plus haut dans le groupe
        max_index = max(clust, key=lambda j: quanti_all_mass[j][2])
        area = quanti_all_mass[max_index][1]

        spec_list.append(spec)
        area_list.append(area)

    return spec_list, area_list


def deconvolution(chromato_cube, time_rn, mod_time, coordinate, radius, multi_processing=True, plot_deconvo=False) :
    inputs = range(chromato_cube.shape[0])
    results = []
    if (multi_processing):
        num_workers = min(multiprocessing.cpu_count(), 32)
        with multiprocessing.Pool(processes=num_workers) as pool:
            for i, result in enumerate(pool.starmap(deconvolution_per_mass, [
                (coordinate[m], radius[m], chromato_cube[m, :, :],
                 time_rn, mod_time, plot_deconvo) for m in inputs
                 ])):
                results.append(result)
    else:
        for m in inputs:
            results.append(deconvolution_per_mass(coordinate[m], radius[m], chromato_cube[m,:,:], time_rn, mod_time, plot_deconvo))
    return results


def deconvolution_per_mass(coordinates, radius, chomato_mass, time_rn, mod_time, plot_deconvo=False):
    if (len(coordinates) > 0):
        coordinates_in_chromato = projection.matrix_to_chromato((coordinates), time_rn, mod_time, chomato_mass.shape)
        bounds = compute_bounds(coordinates_in_chromato, radius)
        clusters = cluster_bounds(coordinates_in_chromato, bounds, time_rn, mod_time, chomato_mass.shape)
        final_bounds = expand_cluster_bounds(bounds, clusters)
        # tic= baseline_correct(chomato_mass,lmbd=1)
        res = []
        for j in range(len(clusters)):
            res.append(fit_all_peaks_fixed_centers(
                chomato_mass, coordinates[clusters[j]],
                projection.chromato_to_matrix(
                    final_bounds[j].T, time_rn, mod_time,
                    chomato_mass.shape).T, plot=plot_deconvo
                    ))

        area = []
        height = []
        coord_order = []
        for x in res:
            for j in x:
                height.append(j['params'][0])
                area.append(j['area'])   
                coord_order.append(j['center'])  

        return np.column_stack((area, height)), np.array(coord_order)
    else:
        return ([], [])
