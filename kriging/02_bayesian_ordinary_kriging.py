"""
Bayesian Ordinary Kriging
==========================
Extends Example 1 by propagating variogram parameter uncertainty through
to grade estimates at unsampled locations. Instead of kriging once with a
single "best-fit" variogram, we krige with many posterior samples and
accumulate a full distribution of estimated grades at each prediction point.

Interview talking point:
  "Classical OK gives you one kriged estimate and one kriging variance.
   Bayesian OK gives you a posterior predictive distribution at each location
   -- you can read off P(grade > cutoff) directly, which feeds straight into
   probabilistic resource classification and cutoff optimisation."

Dependencies: pymc arviz numpy scipy matplotlib
Run AFTER 01_bayesian_variogram.py (re-runs the variogram fit internally).
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.spatial.distance import cdist, pdist, squareform


# ── Pure functions (safe to import in worker processes) ────────────────────

def spherical_cov(dists, nugget, sill, range_):
    h = dists / range_
    return np.where(
        dists == 0, nugget + sill,
        np.where(dists < range_,
                 sill * (1 - 1.5*h + 0.5*h**3), 0.0))


def experimental_variogram(coords, values, n_lags=12, max_dist=None):
    dists_flat = pdist(coords)
    diffs_flat = pdist(values[:, None], metric="sqeuclidean") / 2
    if max_dist is None:
        max_dist = dists_flat.max() / 2
    edges = np.linspace(0, max_dist, n_lags + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    gamma, counts = np.zeros(n_lags), np.zeros(n_lags, dtype=int)
    for k in range(n_lags):
        mask = (dists_flat >= edges[k]) & (dists_flat < edges[k+1])
        counts[k] = mask.sum()
        if counts[k] > 0:
            gamma[k] = diffs_flat[mask].mean()
    valid = counts > 0
    return centers[valid], gamma[valid]


def spherical_variogram_pt(h, nugget, sill, range_):
    import pytensor.tensor as pt
    h_s = h / range_
    return pt.switch(pt.lt(h, 1e-6), 0.0,
           pt.switch(pt.lt(h_s, 1.0),
                     nugget + sill*(1.5*h_s - 0.5*h_s**3),
                     nugget + sill))


def spherical_cov_np(dists, nugget, sill, range_):
    """Vectorised NumPy spherical covariance (not variogram)."""
    h = np.clip(dists / range_, 0, None)
    return np.where(dists < 1e-10,
                    nugget + sill,
                    np.where(h < 1.0,
                             sill*(1 - 1.5*h + 0.5*h**3),
                             0.0))


def ordinary_kriging(coords, values, pred_points, nugget, sill, range_):
    """
    Standard OK with Lagrange multiplier for the unbiasedness constraint.
    Returns kriged estimates and kriging variances at pred_points.

    System (n+1 x n+1):
      [ C   1 ] [ w ]   [ c0 ]
      [ 1T  0 ] [ mu ] = [  1 ]
    """
    n = len(values)
    C_dd = spherical_cov_np(
        squareform(pdist(coords)), nugget, sill, range_)
    C_dd += np.eye(n) * 1e-8

    A = np.zeros((n+1, n+1))
    A[:n, :n] = C_dd
    A[:n,  n] = 1.0
    A[n,  :n] = 1.0

    dist_dp = cdist(coords, pred_points)
    C_dp = spherical_cov_np(dist_dp, nugget, sill, range_)  # (n, n_pred)

    b = np.vstack([C_dp, np.ones((1, len(pred_points)))])   # (n+1, n_pred)

    try:
        W = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A, b, rcond=None)[0]

    weights  = W[:n, :]   # (n, n_pred)
    lagrange = W[n,  :]   # (n_pred,)

    z_hat = weights.T @ values  # (n_pred,)

    c00 = nugget + sill
    sigma2_k = c00 - np.sum(weights * C_dp, axis=0) - lagrange

    return z_hat, np.maximum(sigma2_k, 0.0)


def main():
    # ══════════════════════════════════════════════════════════════════════
    # PART A -- Rebuild the synthetic dataset (identical to Example 1)
    # ══════════════════════════════════════════════════════════════════════
    np.random.seed(42)

    N = 60
    coords = np.random.uniform(0, 400, size=(N, 2))

    D = squareform(pdist(coords))
    TRUE_NUGGET, TRUE_SILL, TRUE_RANGE = 0.05, 0.80, 120.0
    C = spherical_cov(D, TRUE_NUGGET, TRUE_SILL, TRUE_RANGE) + np.eye(N)*1e-6
    log_grades = np.random.multivariate_normal(mean=np.zeros(N), cov=C)
    grades = np.exp(log_grades + 1.2)
    log_z = np.log(grades)   # krige in log space (standard for gold)

    print(f"Dataset: {N} drill holes  |  log-grade variance: {log_z.var():.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # PART B -- Re-run variogram posterior (fast: 1000 draws)
    # ══════════════════════════════════════════════════════════════════════
    lag_h, gamma_exp = experimental_variogram(coords, log_z)

    print("\nFitting variogram posterior ...")
    with pm.Model():
        nugget    = pm.HalfNormal("nugget",    sigma=0.3)
        sill      = pm.HalfNormal("sill",      sigma=log_z.var()*1.5)
        range_    = pm.Uniform("range",        lower=20.0, upper=300.0)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)
        gamma_mod = pm.math.stack([
            spherical_variogram_pt(h, nugget, sill, range_) for h in lag_h])
        pm.Normal("obs", mu=gamma_mod, sigma=sigma_obs, observed=gamma_exp)
        trace = pm.sample(draws=1000, tune=1000, chains=4,
                          target_accept=0.90, random_seed=42,
                          progressbar=True)

    post_nugget = trace.posterior["nugget"].values.flatten()
    post_sill   = trace.posterior["sill"].values.flatten()
    post_range  = trace.posterior["range"].values.flatten()
    print(f"Variogram posterior: {len(post_nugget)} samples available")

    # ══════════════════════════════════════════════════════════════════════
    # PART C -- Prediction grid
    # ══════════════════════════════════════════════════════════════════════
    grid_res = 20   # 20 m grid (increase for finer maps, slower runtime)
    gx = np.arange(0, 401, grid_res)
    gy = np.arange(0, 401, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    pred_pts = np.column_stack([GX.ravel(), GY.ravel()])
    n_pred   = len(pred_pts)
    print(f"\nPrediction grid: {len(gx)}x{len(gy)} = {n_pred} points")

    # ══════════════════════════════════════════════════════════════════════
    # PART D -- Propagate variogram uncertainty through kriging
    # ══════════════════════════════════════════════════════════════════════
    N_SAMPLES = 200   # 200 is fast (~30 s); use 500+ for publication figures

    idx     = np.random.choice(len(post_nugget), N_SAMPLES, replace=False)
    z_hats  = np.zeros((N_SAMPLES, n_pred))
    sigma2s = np.zeros((N_SAMPLES, n_pred))

    print(f"\nKriging with {N_SAMPLES} posterior variogram samples ...")
    for i, j in enumerate(idx):
        if i % 50 == 0:
            print(f"  Sample {i}/{N_SAMPLES}")
        z_hats[i], sigma2s[i] = ordinary_kriging(
            coords, log_z, pred_pts,
            post_nugget[j], post_sill[j], post_range[j])

    # Back-transform: log-normal correction  E[Z] = exp(mu + sigma^2/2)
    grade_mean_log = z_hats.mean(axis=0)
    grade_var_log  = z_hats.var(axis=0) + sigma2s.mean(axis=0)
    grade_mean     = np.exp(grade_mean_log + grade_var_log/2)
    grade_std      = grade_mean * np.sqrt(np.exp(grade_var_log) - 1)

    # P(grade > cutoff) -- directly useful for resource reporting
    CUTOFF_GT = 2.0   # g/t Au economic cutoff
    p_above_cutoff = (z_hats > np.log(CUTOFF_GT)).mean(axis=0)

    print(f"\nKriging complete")
    print(f"   Grade range (back-transformed): "
          f"{grade_mean.min():.2f} - {grade_mean.max():.2f} g/t Au")
    print(f"   Blocks with P(Au > {CUTOFF_GT} g/t) > 0.5: "
          f"{(p_above_cutoff > 0.5).sum()} / {n_pred}")

    # ══════════════════════════════════════════════════════════════════════
    # PART E -- Plot
    # ══════════════════════════════════════════════════════════════════════
    def grid_map(ax, values, title, cmap, label, vmin=None, vmax=None):
        Z = values.reshape(GX.shape)
        im = ax.pcolormesh(GX, GY, Z, cmap=cmap, shading="auto",
                           vmin=vmin, vmax=vmax)
        ax.scatter(coords[:,0], coords[:,1], c="white", s=15, zorder=5,
                   edgecolors="black", linewidths=0.5, label="Drill holes")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Bayesian Ordinary Kriging -- Gold deposit (synthetic)", fontsize=13)

    grid_map(axes[0,0], grade_mean,
             "Posterior mean grade (g/t Au)", "YlOrRd", "g/t Au")

    grid_map(axes[0,1], grade_std,
             "Posterior std dev (g/t Au)", "Blues", "g/t Au std")

    grid_map(axes[1,0], p_above_cutoff,
             f"P(Au > {CUTOFF_GT} g/t) -- ore probability", "RdYlGn",
             "Probability", vmin=0, vmax=1)
    Z_p = p_above_cutoff.reshape(GX.shape)
    axes[1,0].contour(GX, GY, Z_p, levels=[0.5], colors="black",
                      linewidths=1.5, linestyles="--")
    axes[1,0].text(10, 10, "-- P=0.5 boundary", fontsize=8, color="black")

    param_var   = z_hats.var(axis=0)
    kriging_var = sigma2s.mean(axis=0)
    frac_param  = param_var / (param_var + kriging_var + 1e-12)

    grid_map(axes[1,1], frac_param,
             "Fraction of uncertainty from variogram parameters", "PuOr",
             "Fraction [0=kriging, 1=param]", vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig("kriging/bayesian_kriging_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to kriging/bayesian_kriging_results.png")

    # ══════════════════════════════════════════════════════════════════════
    # PART F -- Resource summary table (proto-JORC style)
    # ══════════════════════════════════════════════════════════════════════
    print("\n-- Proto-resource summary -----------------------------------------")
    print(f"  Cutoff: {CUTOFF_GT} g/t Au")
    print(f"  Grid spacing: {grid_res} m  |  Block area: {grid_res**2} m^2")
    print()

    for label, lo, hi in [
        ("High confidence  (P>0.80)", 0.80, 1.01),
        ("Medium           (P 0.5-0.8)", 0.50, 0.80),
        ("Low              (P 0.2-0.5)", 0.20, 0.50),
    ]:
        mask = (p_above_cutoff >= lo) & (p_above_cutoff < hi)
        n_blocks = mask.sum()
        if n_blocks == 0:
            continue
        mean_g = grade_mean[mask].mean()
        area_ha = n_blocks * grid_res**2 / 10_000
        print(f"  {label}")
        print(f"    Blocks: {n_blocks:>4}  |  Area: {area_ha:.1f} ha  "
              f"|  Mean grade: {mean_g:.2f} g/t Au")

    print("""
Interview talking point:
  "The P(grade > cutoff) map is a direct output of the Bayesian posterior
   predictive distribution. Blocks with P > 0.8 map naturally to Indicated
   resources; P 0.5-0.8 to Inferred. This is more honest than a hard
   domain boundary from a single kriged estimate -- the uncertainty is
   explicit and auditable."
""")


if __name__ == "__main__":
    main()
