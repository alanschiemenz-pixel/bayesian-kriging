"""
Bayesian Variogram Estimation
==============================
Fits a spherical variogram model to synthetic drill-hole gold assay data
using PyMC. Instead of least-squares curve fitting, we get full posterior
distributions over nugget, sill, and range -- critical for honest uncertainty
quantification in resource estimation.

Interview talking point:
  "Classical variogram fitting minimises a loss function and gives you one
   answer. Bayesian fitting gives you a posterior distribution over the
   variogram parameters, which propagates through to kriging variance and
   ultimately to resource classification confidence."

Dependencies: pymc arviz numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.spatial.distance import pdist, squareform


# ── Pure functions (safe to import in worker processes) ────────────────────

def spherical_cov(dists, nugget, sill, range_):
    """Spherical covariance model (numpy, for data simulation)."""
    h = dists / range_
    return np.where(
        dists == 0,
        nugget + sill,
        np.where(
            dists < range_,
            sill * (1 - 1.5 * h + 0.5 * h**3),
            0.0,
        ),
    )


def experimental_variogram(coords, values, n_lags=12, max_dist=None):
    """Method-of-moments variogram estimator (Matheron 1963)."""
    dists_flat = pdist(coords)
    diffs_flat = pdist(values[:, None], metric="sqeuclidean") / 2  # gamma = 0.5*(z_i-z_j)^2

    if max_dist is None:
        max_dist = dists_flat.max() / 2  # rule of thumb: half the max distance

    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        mask = (dists_flat >= lag_edges[k]) & (dists_flat < lag_edges[k + 1])
        counts[k] = mask.sum()
        if counts[k] > 0:
            gamma[k] = diffs_flat[mask].mean()

    valid = counts > 0
    return lag_centers[valid], gamma[valid], counts[valid]


def spherical_variogram_pt(h, nugget, sill, range_):
    """Spherical variogram model in pytensor (used inside PyMC model)."""
    import pytensor.tensor as pt
    h_scaled = h / range_
    return pt.switch(
        pt.lt(h, 1e-6),
        0.0,
        pt.switch(
            pt.lt(h_scaled, 1.0),
            nugget + sill * (1.5 * h_scaled - 0.5 * h_scaled**3),
            nugget + sill,
        ),
    )


def main():
    # ── 1. Simulate synthetic drill-hole data ──────────────────────────────
    # True variogram: spherical, nugget=0.05, sill=0.8, range=120 m
    np.random.seed(41)
    N = 60
    TRUE_NUGGET, TRUE_SILL, TRUE_RANGE = 0.05, 0.80, 120.0

    coords = np.random.uniform(0, 400, size=(N, 2))

    D = squareform(pdist(coords))
    C = spherical_cov(D, TRUE_NUGGET, TRUE_SILL, TRUE_RANGE)
    C += np.eye(N) * 1e-6  # numerical stability

    log_grades = np.random.multivariate_normal(mean=np.zeros(N), cov=C)
    grades = np.exp(log_grades + 1.2)  # shift so mean ~3.3 g/t Au

    print(f"Simulated {N} intercepts  |  grade range: "
          f"{grades.min():.2f} - {grades.max():.2f} g/t Au")

    # ── 2. Compute experimental variogram ──────────────────────────────────
    lag_h, gamma_exp, n_pairs = experimental_variogram(coords, np.log(grades))

    print(f"\nExperimental variogram computed over {len(lag_h)} lag classes")
    print(f"Max lag used: {lag_h.max():.0f} m  |  pairs per bin: "
          f"{n_pairs.min()} - {n_pairs.max()}")

    # ── 3. Bayesian variogram model ────────────────────────────────────────
    # Model: observed gamma(h) ~ Normal(gamma_spherical(h | theta), sigma_obs)
    # Priors:
    #   nugget    ~ HalfNormal  (assay error + micro-scale variability)
    #   sill      ~ HalfNormal  (total variance of the field)
    #   range     ~ Uniform     (plausible deposit-scale continuity)
    #   sigma_obs ~ HalfNormal  (scatter of MoM estimator)
    emp_var = float(np.var(np.log(grades)))

    with pm.Model() as vario_model:
        nugget    = pm.HalfNormal("nugget", sigma=0.3)
        sill      = pm.HalfNormal("sill", sigma=emp_var * 1.5)
        range_    = pm.Uniform("range", lower=20.0, upper=300.0)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)

        gamma_model = pm.math.stack([
            spherical_variogram_pt(h, nugget, sill, range_) for h in lag_h
        ])

        pm.Normal("obs", mu=gamma_model, sigma=sigma_obs, observed=gamma_exp)

    print("\n-- Model structure ------------------------------------------------")
    print(vario_model.debug())

    # ── 4. Sample the posterior (MCMC) ────────────────────────────────────
    print("\n-- Sampling -------------------------------------------------------")
    with vario_model:
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.90,
            random_seed=42,
            progressbar=True,
        )

    # ── 5. Convergence diagnostics ─────────────────────────────────────────
    print("\n-- Convergence diagnostics ----------------------------------------")
    summary = az.summary(trace, var_names=["nugget", "sill", "range", "sigma_obs"])
    print(summary)
    # R-hat < 1.01 and ESS_bulk > 400 per chain are the targets

    rhat_max = summary["r_hat"].max()
    if rhat_max > 1.05:
        print(f"\nWARNING: Max R-hat = {rhat_max:.3f} -- chains may not have converged!")
        print("   Try: more tuning steps, reparametrisation, or informative priors.")
    else:
        print(f"\nOK: Max R-hat = {rhat_max:.3f} -- good convergence.")

    # ── 6. Plot results ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle("Bayesian Variogram Estimation -- Gold deposit (synthetic)", fontsize=13)

    # Panel A: Posterior variogram fan (top row, spans first 2 columns)
    ax_fan = fig.add_subplot(gs[0, :2])
    h_plot = np.linspace(0.1, lag_h.max() * 1.1, 200)
    post_nugget = trace.posterior["nugget"].values.flatten()
    post_sill   = trace.posterior["sill"].values.flatten()
    post_range  = trace.posterior["range"].values.flatten()

    # 200 random posterior draws as semi-transparent lines
    idx = np.random.choice(len(post_nugget), 200, replace=False)
    for i in idx:
        n, s, r = post_nugget[i], post_sill[i], post_range[i]
        h_s = h_plot / r
        gv = np.where(h_plot < r, n + s * (1.5 * h_s - 0.5 * h_s**3), n + s)
        ax_fan.plot(h_plot, gv, color="#378ADD", alpha=0.04, lw=0.8)

    # Posterior mean curve
    gv_mean = np.zeros_like(h_plot)
    for i in range(len(post_nugget)):
        n, s, r = post_nugget[i], post_sill[i], post_range[i]
        h_s = h_plot / r
        gv_mean += np.where(h_plot < r, n + s * (1.5 * h_s - 0.5 * h_s**3), n + s)
    gv_mean /= len(post_nugget)
    ax_fan.plot(h_plot, gv_mean, color="#1d4ed8", lw=2, label="Posterior mean")
    ax_fan.scatter(lag_h, gamma_exp, c="black", zorder=5, s=30,
                   label=f"Experimental (n={len(lag_h)} lags)")
    ax_fan.set_xlabel("Lag distance h (m)")
    ax_fan.set_ylabel("gamma(h)")
    ax_fan.set_title("Variogram posterior fan")
    ax_fan.legend(fontsize=9)

    # Panel B: Trace plot for range (top-right)
    ax_trace = fig.add_subplot(gs[0, 2])
    n_chains = trace.posterior.sizes["chain"]
    for chain in range(n_chains):
        ax_trace.plot(trace.posterior["range"].values[chain],
                      alpha=0.7, lw=0.6, label=f"Chain {chain}")
    ax_trace.axhline(TRUE_RANGE, color="red", ls="--", lw=1.5,
                     label=f"True range={TRUE_RANGE} m")
    ax_trace.set_xlabel("Draw")
    ax_trace.set_ylabel("Range (m)")
    ax_trace.set_title("Trace plot -- range parameter")
    ax_trace.legend(fontsize=8)

    # Panel C: Marginal posteriors (bottom row, one axes per parameter)
    ax_nug  = fig.add_subplot(gs[1, 0])
    ax_sil  = fig.add_subplot(gs[1, 1])
    ax_rng  = fig.add_subplot(gs[1, 2])
    az.plot_posterior(trace, var_names=["nugget"], ax=ax_nug,
                      kind="hist", color="#378ADD", textsize=9)
    az.plot_posterior(trace, var_names=["sill"],   ax=ax_sil,
                      kind="hist", color="#378ADD", textsize=9)
    az.plot_posterior(trace, var_names=["range"],  ax=ax_rng,
                      kind="hist", color="#378ADD", textsize=9)

    plt.savefig("kriging/bayesian_variogram_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to kriging/bayesian_variogram_results.png")

    # ── 7. Summary output ──────────────────────────────────────────────────
    print("\n-- Posterior summary (95% HDI) ------------------------------------")
    for param, true_val in [("nugget", TRUE_NUGGET), ("sill", TRUE_SILL), ("range", TRUE_RANGE)]:
        vals = trace.posterior[param].values.flatten()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        mean = vals.mean()
        print(f"  {param:<8}  true={true_val:.2f}  |  posterior mean={mean:.2f}  "
              f"  95% HDI=[{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
