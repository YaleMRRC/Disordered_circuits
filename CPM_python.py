import os
import numpy as np
from scipy import io as sio
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from scipy import stats

def _vectorize_lower_triangle(mat_2d):
    """
    Convert symmetric NxN matrix to condensed (lower-tri, excluding diagonal) vector
    matching MATLAB squareform(tril(M, -1)).
    Uses scipy.spatial.distance.squareform with checks=False.
    """
    # squareform will read the lower triangle by default for condensed vector
    # when given a symmetric matrix with zeros on the diagonal.
    return squareform(mat_2d, checks=False)


def _partial_corr_1cov(x, y, z):
    """
    Partial correlation between x and y controlling for one covariate z.
    All are 1D numpy arrays of same length (n,). NaNs are handled by listwise deletion.
    Returns: r, p  (from Pearson correlation on residuals)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
    if mask.sum() < 3:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Residualize x ~ z, y ~ z (simple linear regression with 1 covariate)
    z1 = np.column_stack([np.ones_like(z), z])
    # Solve least squares
    bx, *_ = np.linalg.lstsq(z1, x, rcond=None)
    by, *_ = np.linalg.lstsq(z1, y, rcond=None)

    x_res = x - z1 @ bx
    y_res = y - z1 @ by

    r, p = pearsonr(x_res, y_res)
    return r, p


def _corr_no_nan(x, y):
    """
    Pearson correlation handling NaNs via listwise deletion (rows='complete' in MATLAB).
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])


def _spearman_no_nan(x, y):
    """
    Spearman correlation handling NaNs via listwise deletion (rows='complete').
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    return spearmanr(x[mask], y[mask])


def _vectorized_pearson_all_edges(train_x, train_y):
    """
    Vectorized Pearson correlation between each row of train_x (n_edges x n_subj)
    and train_y (n_subj,). Returns r (n_edges,), p (n_edges,).
    Assumes you already filtered NaNs out of train_y; train_x should normally be NaN-free.
    """
    X = np.asarray(train_x, float)            # (n_edges, n_subj)
    y = np.asarray(train_y, float).reshape(1, -1)  # (1, n_subj)

    # Center
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)   # (1, n_subj)

    # Numerator = covariance * (n-1)
    num = (Xc @ yc.T).ravel()                # (n_edges,)

    # Denominator
    denom = np.sqrt((Xc**2).sum(axis=1)) * np.sqrt((yc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom

    # Degrees of freedom
    n = X.shape[1]
    df = n - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(df / (1.0 - r**2))
    p = 2.0 * stats.t.sf(np.abs(t), df)

    return r, p


def _vectorized_partial_corr_all_edges(train_x, train_y, covars):
    """
    Vectorized partial correlation between each row of train_x (n_edges x n_subj)
    and train_y (n_subj,), controlling jointly for one or more covariates.

    Parameters
    ----------
    train_x : array, shape (n_edges, n_subj)
    train_y : array, shape (n_subj,)
    covars  : array, shape (n_subj, n_covars) or (n_subj,)

    Returns
    -------
    r : array, shape (n_edges,)
        Partial correlation coefficient for each edge.
    p : array, shape (n_edges,)
        Two-sided p-value for each edge.
    """
    X = np.asarray(train_x, float)                  # (n_edges, n_subj)
    y = np.asarray(train_y, float).reshape(-1)      # (n_subj,)
    Z = np.asarray(covars, float)

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)                        # (n_subj, 1)

    # Listwise deletion across y + all covariates
    mask = (~np.isnan(y)) & (~np.isnan(Z).any(axis=1))
    X = X[:, mask]                                  # (n_edges, n_valid)
    y = y[mask]                                     # (n_valid,)
    Z = Z[mask, :]                                  # (n_valid, n_covars)

    n = y.size
    k = Z.shape[1]                                  # number of covariates
    if n < (k + 3):
        return (np.full(X.shape[0], np.nan),
                np.full(X.shape[0], np.nan))

    # Add intercept
    Z = np.column_stack([np.ones(n), Z])            # (n_valid, k+1)

    # Residual maker components
    ZTZ_inv = np.linalg.pinv(Z.T @ Z)               # safer than inv
    G = ZTZ_inv @ Z.T                               # ((k+1), n_valid)

    # Residualize y
    beta_y = G @ y                                  # (k+1,)
    y_res = y - Z @ beta_y                          # (n_valid,)

    # Residualize all edges at once
    X_T = X.T                                       # (n_valid, n_edges)
    beta_X = G @ X_T                                # ((k+1), n_edges)
    X_res = X_T - Z @ beta_X                        # (n_valid, n_edges)

    # Correlate residualized y with each residualized edge
    yrc = y_res - y_res.mean()
    Xrc = X_res - X_res.mean(axis=0, keepdims=True)

    num = yrc @ Xrc                                 # (n_edges,)
    denom = np.sqrt((yrc ** 2).sum()) * np.sqrt((Xrc ** 2).sum(axis=0))

    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom

    # df for partial correlation with k covariates
    df = n - k - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(df / (1.0 - r ** 2))
    p = 2.0 * stats.t.sf(np.abs(t), df)

    return r, p

def runCPM(x, y, kfolds, age=None, sex=None, random_state=42):
    """
    Faster CPM: vectorized edgewise correlations / partial correlations.
    Same inputs/outputs as your original runCPM.
    """
    N = len(y)
    n_nodes = x.shape[0]
    n_edges = n_nodes * (n_nodes - 1) // 2

    # --- Vectorize all subjects once ---
    all_edges = np.zeros((n_edges, N), dtype=float)
    for s in range(N):
        mat = x[:, :, s].copy()
        np.fill_diagonal(mat, 0.0)
        all_edges[:, s] = _vectorize_lower_triangle(mat)

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)

    all_pos_edges = np.zeros((n_edges, kfolds), dtype=bool)
    all_neg_edges = np.zeros((n_edges, kfolds), dtype=bool)
    pred_Y = np.full(N, np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):
        train_x = all_edges[:, train_idx]   # (n_edges, Ntrain)
        test_x  = all_edges[:, test_idx]    # (n_edges, Ntest)

        train_y = y[train_idx]
        test_y  = y[test_idx]  # unused but kept for symmetry

        train_age = age[train_idx] if age is not None else None
        train_sex = sex[train_idx] if sex is not None else None

        # --- Univariate edge selection ---
        if age is not None or sex is not None:
            cov_list = []
            if train_age is not None:
                cov_list.append(train_age)
            if train_sex is not None:
                cov_list.append(train_sex)

            covars = np.column_stack(cov_list)
            r, p = _vectorized_partial_corr_all_edges(train_x, train_y, covars)

            edges_pos = (p < 0.05) & (r > 0)
            edges_neg = (p < 0.05) & (r < 0)

        else:
            r, p = _vectorized_pearson_all_edges(train_x, train_y)
            edges_pos = (p < 0.05) & (r > 0)
            edges_neg = (p < 0.05) & (r < 0)

        all_pos_edges[:, fold_idx] = edges_pos
        all_neg_edges[:, fold_idx] = edges_neg

        # --- Build model on TRAIN ---
        train_sum = np.nansum(train_x[edges_pos, :], axis=0) - np.nansum(
            train_x[edges_neg, :], axis=0
        )

        if np.all(np.isnan(train_sum)) or np.isnan(train_y).all():
            a, b = np.nan, np.nan
        else:
            m = (~np.isnan(train_sum)) & (~np.isnan(train_y))
            if m.sum() < 2:
                a, b = np.nan, np.nan
            else:
                a, b = np.polyfit(train_sum[m], train_y[m], 1)

        # --- Predict on TEST ---
        test_sum = np.sum(test_x[edges_pos, :], axis=0) - np.sum(
            test_x[edges_neg, :], axis=0
        )
        if np.isnan(a) or np.isnan(b):
            yhat = np.full(test_sum.shape, np.nan)
        else:
            yhat = a * test_sum + b

        pred_Y[test_idx] = yhat

    # --- Evaluate performance ---
    r_pearson, p_pearson = _corr_no_nan(pred_Y, y)
    r_rank, p_rank = _spearman_no_nan(pred_Y, y)

    mse = np.nanmean((pred_Y - y) ** 2)
    var_y = np.nanvar(y, ddof=0)
    q_s = 1.0 - (mse / var_y) if np.isfinite(var_y) and var_y > 0 else np.nan

    stats = {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_rank': r_rank,
        'p_rank': p_rank,
        'mse': mse,
        'q_s': q_s,
    }

    return stats, all_pos_edges, all_neg_edges, pred_Y

def main(
    averaged_mats_path='/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/averaged_mats.mat',
    behdata_path='/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/BSI_items.xlsx',
    confounds_path='/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/confounds.mat',
    outfile='/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/items_test_predictions.mat',
    nperms=1,
    kfolds=2,
    edge_thresh=0.5,
    random_state=42,
    t_indices=None
):
    # Load data
    mats = sio.loadmat(averaged_mats_path, squeeze_me=True, struct_as_record=False)
    avg_mats = mats.get('avg_mats')
    if avg_mats is None:
        raise ValueError("avg_mats not found in averaged_mats.mat")
    # Expect shape (Nnodes, Nnodes, Nsubjects)
    if avg_mats.ndim != 3:
        raise ValueError(f"avg_mats must be 3D; got shape {avg_mats.shape}")

    beh = sio.loadmat(behdata_path, squeeze_me=True, struct_as_record=False)
    behdata = beh.get('behdata')
    behheader = beh.get('behheader')
    if behdata is None or behheader is None:
        raise ValueError("behdata and/or behheader not found in behdata_to_model.mat")

    # Normalize headers to list of strings
    if isinstance(behheader, np.ndarray):
        behheader = [str(h) for h in behheader.tolist()]
    elif isinstance(behheader, (list, tuple)):
        behheader = [str(h) for h in behheader]
    else:
        behheader = [str(behheader)]

    conf = sio.loadmat(confounds_path, squeeze_me=True, struct_as_record=False)
    Confounds = conf.get('Confounds')
    if Confounds is None:
        raise ValueError("Confounds not found in confounds.mat")

    # Prepare outputs
    N = behdata.shape[0]          # subjects (rows)
    nmeas_total = behdata.shape[1]  # total measures (cols)

    # Decide which measures this job should run
    if t_indices is None:
        # Default: run ALL measures
        t_indices = list(range(nmeas_total))
    else:
        # Ensure it's a list of ints
        t_indices = [int(t) for t in t_indices]

    nmeas = len(t_indices)

    # These will be filled per measure t
    p_vals = np.full((nmeas, 1), np.nan)
    predictions = np.full((nperms, nmeas), np.nan)
    null_predictions = np.full((nperms, nmeas), np.nan)

    networks = {}  # will store keys like "<header>_pos", "<header>_neg", "<header>_both"

    # Edge count for threshold
    nedgethresh = int(edge_thresh * kfolds * nperms)


    for local_i, t in enumerate(t_indices):
        header_t = behheader[t]
        print(f"\n=== Measure {t} ({header_t}) ===")

        # Gather non-NaN subjects for this measure
        feat_col = behdata[:, t]  # likely an object/cell-like array

        # Convert to numeric vector (cell2mat equivalent)
        if isinstance(feat_col, np.ndarray) and feat_col.dtype == object:
            y_list = []
            for v in feat_col:
                if v is None:
                    y_list.append(np.nan)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    y_list.append(float(v))
                elif isinstance(v, np.ndarray):
                    # empty cell → treat as missing
                    if v.size == 0:
                        y_list.append(np.nan)
                    else:
                        # non-empty array → convert first element
                        y_list.append(float(np.ravel(v)[0]))
                else:
                    # fallback: try converting, otherwise NaN
                    try:
                        y_list.append(float(v))
                    except:
                        y_list.append(np.nan)

            y_vec = np.array(y_list, dtype=float)
        else:
            y_vec = feat_col.astype(float).copy()

        # Filter subjects with non-NaN y
        keep = ~np.isnan(y_vec)
        ix = keep.sum()
        if ix < 3:
            print(f"Skipping {header_t}: fewer than 3 subjects with non-NaN y.")
            continue

        x = avg_mats[:, :, keep]
        y = y_vec[keep]

        # Format confounds (same as you already had)
        Age = np.zeros(ix, dtype=float)
        Sex = np.zeros(ix, dtype=float)
        keep_idx = np.where(keep)[0]
        for j, orig_i in enumerate(keep_idx):
            row = Confounds[orig_i]
            age_val = row[0]
            try:
                Age[j] = float(age_val) if np.size(age_val) == 1 else float(np.ravel(age_val)[0])
            except Exception:
                Age[j] = np.nan
            sex_str = str(row[1])
            Sex[j] = 1.0 if ('M' in sex_str or 'm' in sex_str) else 2.0

        # ---- Permutation loop ----
        Pred_strength = np.full((nperms, 1), np.nan)
        Null_preds = np.full((nperms, 1), np.nan)

        pos_edges_accum = []
        neg_edges_accum = []

        rng = np.random.default_rng(random_state)

        for np_i in range(nperms):
            if np_i == 0:
                print("Iteration ", end="", flush=True)
            print(f"{np_i+1} ", end="", flush=True)

            stats, all_pos_edges, all_neg_edges, _ = runCPM(
                x, y, kfolds, age=Age, sex=Sex, random_state=random_state + np_i
            )
            Pred_strength[np_i, 0] = stats['r_rank']

            pos_edges_accum.append(all_pos_edges)
            neg_edges_accum.append(all_neg_edges)

            # Null model: shuffle y
            shuffled_idx = rng.permutation(len(y))
            ShuffledY = y[shuffled_idx]

            nullstats, _, _, _ = runCPM(
                x, ShuffledY, kfolds, age=Age, sex=Sex, random_state=random_state + 10_000 + np_i
            )
            Null_preds[np_i, 0] = nullstats['r_rank']

        print("")  # newline after iterations

        pos_edges = np.hstack(pos_edges_accum) if len(pos_edges_accum) else np.empty((0, 0), dtype=bool)
        neg_edges = np.hstack(neg_edges_accum) if len(neg_edges_accum) else np.empty((0, 0), dtype=bool)

        # p-value as count of null > median(real)
        med_pred = np.nanmedian(Pred_strength[:, 0])
        p_val = np.sum(Null_preds[:, 0] > med_pred)
        p_vals[local_i, 0] = p_val
        predictions[:, local_i] = Pred_strength[:, 0]
        null_predictions[:, local_i] = Null_preds[:, 0]

        # ---- Compute network score (unchanged logic, just uses local_i / header_t) ----
        all_pos_counts = np.sum(pos_edges, axis=1)
        all_neg_counts = np.sum(neg_edges, axis=1)

        pos_pred_edges = np.where(all_pos_counts > nedgethresh)[0]
        neg_pred_edges = np.where(all_neg_counts > nedgethresh)[0]

        n_edges = all_pos_counts.shape[0]
        pos_network = np.zeros((n_edges, 1), dtype=int)
        neg_network = np.zeros((n_edges, 1), dtype=int)
        pos_network[pos_pred_edges, 0] = 1
        neg_network[neg_pred_edges, 0] = 1

        networks[f"{header_t}_pos"] = pos_network
        networks[f"{header_t}_neg"] = neg_network
        networks[f"{header_t}_both"] = pos_network + neg_network

        # ---- netscores ----
        n_kept = x.shape[2]
        edge_vecs = np.zeros((n_edges, n_kept), dtype=float)
        for s in range(n_kept):
            mat = x[:, :, s].copy()
            np.fill_diagonal(mat, 0.0)
            edge_vecs[:, s] = _vectorize_lower_triangle(mat)

        netscores_t = np.sum(edge_vecs[pos_pred_edges, :], axis=0) - np.sum(
            edge_vecs[neg_pred_edges, :], axis=0
        )

        networks[f"{header_t}_netscores_idx"] = keep_idx
        networks[f"{header_t}_netscores"] = netscores_t.reshape(-1, 1)

    # Save results to .mat 
    behheader_subset = np.array([behheader[t] for t in t_indices], dtype=object)

    outdict = {
        'p_vals': p_vals,
        'predictions': predictions,
        'null_predictions': null_predictions,
        'networks': networks,
        'behheader_subset': behheader_subset,
        'measure_indices': np.array(t_indices, dtype=int),
    }
    
    sio.savemat(outfile, outdict)
    print(f"\nSaved results to: {outfile}")

if __name__ == "__main__":
    meas_env = os.environ.get("MEAS_INDEX", "").strip()
    t_indices = None
    if meas_env:
        idxs = []
        for part in meas_env.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                idxs.extend(list(range(int(a), int(b) + 1)))
            else:
                idxs.append(int(part))
        t_indices = idxs

    main(
        averaged_mats_path=os.environ.get("AVERAGED_MATS_PATH", '/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/averaged_mats.mat'),
        behdata_path=os.environ.get("BEHDATA_PATH", "/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/BSI_items.mat"),
        confounds_path=os.environ.get("CONFOUNDS_PATH", '/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/CPM_input_data/confounds.mat'),
        outfile=os.environ.get("OUTFILE", '/Users/ajsimon/Documents/Data/Constable_lab/Transdiagnostic/N317/items_test_predictions.mat'),
        nperms=int(os.environ.get("NPERMS", "1")),
        kfolds=int(os.environ.get("KFOLDS", "2")),
        edge_thresh=float(os.environ.get("EDGE_THRESH", "0.5")),
        random_state=int(os.environ.get("RANDOM_STATE", "42")),
        t_indices=t_indices,
    )
