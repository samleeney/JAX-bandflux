"""
Nested sampling anomaly detection example using SALT3 with structured priors.

This script mirrors the ``ns.py`` configuration style:
    - Parameters live in dictionaries compatible with JAX PyTrees
    - Uniform priors are generated via ``blackjax.ns.utils.uniform_prior``
    - Log-likelihoods are JIT-compiled and operate on structured inputs

On top of the standard SALT3 fit it includes an anomaly model with a ``log_p``
parameter that can down-weight outlying photometric points. Both runs are
compared via corner plots and a light-curve diagnostic that highlights likely
anomalies.
"""

import os
import anesthetic
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from anesthetic import make_2d_axes
from blackjax.ns.utils import finalise, uniform_prior, log_weights
from jax.scipy.special import logsumexp
from jax_supernovae import SALT3Source
from jax_supernovae.data import load_and_process_data

# Enable float64 precision for the model and nested sampling machinery.
jax.config.update("jax_enable_x64", True)

# Configuration constants
SUPERNOVA_ID = "19dwz"
DATA_DIR = "data"
FIXED_Z = 0.04607963148708845
NS_SETTINGS = {
    "n_delete": 60,
    "n_live": 125,
    "num_mcmc_steps_multiplier": 5,
    "max_iterations": 500,
}

# Prior bounds: sample SALT3 parameters uniformly in log_x0 rather than x0.
PRIOR_BOUNDS_STANDARD = {
    "t0": (58000.0, 59000.0),
    "log_x0": (-5.0, -2.6),
    "x1": (-4.0, 4.0),
    "c": (-0.3, 0.3),
}

PRIOR_BOUNDS_ANOMALY = {
    **PRIOR_BOUNDS_STANDARD,
    "log_p": (-20.0, -1.0),
}

STANDARD_PARAM_NAMES = list(PRIOR_BOUNDS_STANDARD.keys())
ANOMALY_PARAM_NAMES = list(PRIOR_BOUNDS_ANOMALY.keys())
NUM_MCMC_STEPS_STANDARD = len(STANDARD_PARAM_NAMES) * NS_SETTINGS["num_mcmc_steps_multiplier"]
NUM_MCMC_STEPS_ANOMALY = len(ANOMALY_PARAM_NAMES) * NS_SETTINGS["num_mcmc_steps_multiplier"]

OUTPUT_DIR = f"chains_{SUPERNOVA_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess the photometric data.
(
    times,
    fluxes,
    fluxerrs,
    zps,
    band_indices,
    unique_bands,
    bridges,
    _,
) = load_and_process_data(
    SUPERNOVA_ID,
    data_dir=DATA_DIR,
    fix_z=False,
)

# Instantiate SALT3 source for bandflux calls.
source = SALT3Source()

# Precompute constants used in the likelihoods.
LOG_DET = jnp.sum(jnp.log(2.0 * jnp.pi * fluxerrs**2))
ANOMALY_NORMALISATION = -0.5 * jnp.log(2.0 * jnp.pi * fluxerrs**2)
ANOMALY_DELTA = jnp.max(jnp.abs(fluxes))


@jax.jit
def _loglikelihood_standard_single(params: dict) -> jnp.ndarray:
    """Return log-likelihood for a single set of standard SALT3 parameters."""
    t0 = params["t0"]
    log_x0 = params["log_x0"]
    x1 = params["x1"]
    c = params["c"]

    x0 = 10.0**log_x0
    phases = (times - t0) / (1.0 + FIXED_Z)
    param_dict = {"x0": x0, "x1": x1, "c": c}

    model_fluxes = source.bandflux(
        param_dict,
        bands=None,
        phases=phases,
        zp=zps,
        zpsys="ab",
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands,
    )

    residuals = (fluxes - model_fluxes) / fluxerrs
    chi2 = jnp.sum(residuals**2)
    return -0.5 * (chi2 + LOG_DET)


@jax.jit
def loglikelihood_standard(params: dict) -> jnp.ndarray:
    """Vectorised log-likelihood for the standard SALT3 model."""
    return jax.vmap(_loglikelihood_standard_single)(params)


@jax.jit
def _loglikelihood_anomaly_single(params: dict):
    """Log-likelihood and anomaly mask for a single parameter set."""
    t0 = params["t0"]
    log_x0 = params["log_x0"]
    x1 = params["x1"]
    c = params["c"]
    log_p = params["log_p"]

    x0 = 10.0**log_x0
    p = jnp.exp(log_p)
    phases = (times - t0) / (1.0 + FIXED_Z)
    param_dict = {"x0": x0, "x1": x1, "c": c}

    model_fluxes = source.bandflux(
        param_dict,
        bands=None,
        phases=phases,
        zp=zps,
        zpsys="ab",
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands,
    )

    residuals = (fluxes - model_fluxes) / fluxerrs
    point_logL = -0.5 * residuals**2 + ANOMALY_NORMALISATION + jnp.log1p(-p)
    log_floor = log_p - jnp.log(ANOMALY_DELTA)
    emax = point_logL > log_floor
    logL = jnp.where(emax, point_logL, log_floor)
    return jnp.sum(logL), emax


@jax.jit
def loglikelihood_anomaly(params: dict) -> jnp.ndarray:
    """Vectorised anomaly log-likelihood (returns only logL)."""
    logL, _ = jax.vmap(_loglikelihood_anomaly_single)(params)
    return logL


@jax.jit
def anomaly_loglikelihood_with_mask(params: dict):
    """Vectorised anomaly log-likelihood returning (logL, emax)."""
    return jax.vmap(_loglikelihood_anomaly_single)(params)


def run_nested_sampling(
    prior_bounds: dict,
    param_names: list[str],
    loglikelihood_fn,
    label: str,
    num_mcmc_steps: int,
):
    """Run a nested sampling configuration and save the resulting samples."""
    print(f"\nSetting up {label} nested sampling...")

    rng_key = jax.random.PRNGKey(0)
    rng_key, prior_key = jax.random.split(rng_key)
    particles, logprior_fn = uniform_prior(prior_key, NS_SETTINGS["n_live"], prior_bounds)

    print(f"Particle structure: {particles.keys()}")
    print(f"Shape of each parameter: {particles[param_names[0]].shape}")

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=num_mcmc_steps,
        num_delete=NS_SETTINGS["n_delete"],
    )

    state = algo.init(particles)
    print("Using device:", jax.devices()[0])

    @jax.jit
    def one_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, dead_point = algo.step(subkey, state)
        return (state, key), dead_point

    dead_points = []
    iterations = 0
    with tqdm.tqdm(desc=f"{label.capitalize()} dead points", unit=" dead") as progress:
        while not state.logZ_live - state.logZ < -3.0 and iterations < NS_SETTINGS["max_iterations"]:
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead_points.append(dead_info)
            progress.update(NS_SETTINGS["n_delete"])
            iterations += 1

    if iterations >= NS_SETTINGS["max_iterations"]:
        print(f"Warning: reached max_iterations={NS_SETTINGS['max_iterations']} before convergence.")

    ns_run = finalise(state, dead_points)

    nested_samples = anesthetic.NestedSamples(
        data={name: ns_run.particles[name] for name in param_names},
        logL=ns_run.logL,
        logL_birth=ns_run.logL_birth,
    )

    csv_path = os.path.join(OUTPUT_DIR, f"{label}_samples.csv")
    nested_samples.to_csv(csv_path)
    print(f"Saved samples to {csv_path}")

    return ns_run, nested_samples


def compute_weighted_emax(ns_run) -> np.ndarray:
    """Compute weighted anomaly mask expectations for the anomaly run."""
    logw = log_weights(jax.random.PRNGKey(1), ns_run)
    logw_mean = logw.mean(axis=-1)
    normalised_logw = logw_mean - logsumexp(logw_mean)
    weights = jnp.exp(normalised_logw)

    _, emax = anomaly_loglikelihood_with_mask(ns_run.particles)
    weighted_emax = jnp.sum(emax * weights[:, None], axis=0) / jnp.sum(weights)
    return np.array(weighted_emax)


def create_corner_plots(standard_samples, anomaly_samples):
    """Generate comparison corner plots."""
    print("Creating corner plots...")
    comparison_axes = make_2d_axes(STANDARD_PARAM_NAMES, figsize=(10, 10), facecolor="w")
    standard_samples.plot_2d(comparison_axes, alpha=0.7, label="Standard")
    anomaly_samples[STANDARD_PARAM_NAMES].plot_2d(comparison_axes, alpha=0.7, label="Anomaly")
    comparison_axes.iloc[-1, 0].legend(
        bbox_to_anchor=(len(comparison_axes) / 2, len(comparison_axes)),
        loc="lower center",
        ncol=2,
    )
    comparison_axes.figure.suptitle(
        "SALT3 Parameter Posteriors: Standard vs Anomaly",
        y=1.02,
        fontsize=14,
    )
    comparison_axes.figure.savefig(
        os.path.join(OUTPUT_DIR, "corner_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )

    anomaly_axes = make_2d_axes(ANOMALY_PARAM_NAMES, figsize=(12, 12), facecolor="w")
    anomaly_samples.plot_2d(anomaly_axes, alpha=0.7, label="Anomaly")
    anomaly_axes.figure.suptitle(
        "Anomaly Detection Parameters (including log_p)",
        y=1.02,
        fontsize=14,
    )
    anomaly_axes.figure.savefig(
        os.path.join(OUTPUT_DIR, "corner_anomaly_logp.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(comparison_axes.figure)
    plt.close(anomaly_axes.figure)


def create_light_curve_plot(anomaly_samples, weighted_emax: np.ndarray):
    """Plot the light curve with anomaly indicators."""
    print("Creating light curve plot with anomaly detection...")
    try:
        t0_med = float(anomaly_samples["t0"].median())
        log_x0_med = float(anomaly_samples["log_x0"].median())
        x1_med = float(anomaly_samples["x1"].median())
        c_med = float(anomaly_samples["c"].median())

        median_params = {
            "t0": t0_med,
            "x0": 10.0**log_x0_med,
            "x1": x1_med,
            "c": c_med,
        }

        times_np = np.array(times)
        fluxes_np = np.array(fluxes)
        fluxerrs_np = np.array(fluxerrs)
        zps_np = np.array(zps)
        band_indices_np = np.array(band_indices)

        t_grid = np.linspace(times_np.min() - 10.0, times_np.max() + 10.0, 200)
        phases_grid = (t_grid - median_params["t0"]) / (1.0 + FIXED_Z)

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        colors = ["g", "orange", "r", "brown", "purple", "blue"]
        markers = ["o", "s", "D", "^", "v", "P"]
        threshold = 0.2

        weighted_emax_np = np.array(weighted_emax)

        for i, band_name in enumerate(unique_bands):
            mask = band_indices_np == i
            if not np.any(mask):
                continue

            band_times = times_np[mask]
            band_fluxes = fluxes_np[mask]
            band_errors = fluxerrs_np[mask]
            band_emax = weighted_emax_np[mask]

            normal_mask = band_emax >= threshold
            anomaly_mask = ~normal_mask

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            if np.any(normal_mask):
                ax1.errorbar(
                    band_times[normal_mask],
                    band_fluxes[normal_mask],
                    yerr=band_errors[normal_mask],
                    fmt=marker,
                    color=color,
                    label=f"{band_name}",
                    markersize=6,
                    alpha=0.6,
                )

            if np.any(anomaly_mask):
                ax1.errorbar(
                    band_times[anomaly_mask],
                    band_fluxes[anomaly_mask],
                    yerr=band_errors[anomaly_mask],
                    fmt="*",
                    color=color,
                    markersize=12,
                    alpha=0.8,
                )

            model_fluxes_grid = source.bandflux(
                {"x0": median_params["x0"], "x1": median_params["x1"], "c": median_params["c"]},
                band_name,
                phases_grid,
                zp=zps_np[mask][0] if np.any(mask) else zps_np[0],
                zpsys="ab",
            )
            ax1.plot(t_grid, model_fluxes_grid, "-", color=color, linewidth=2, alpha=0.7)

        ax1.set_ylabel("Flux", fontsize=12)
        ax1.set_title(f"Light Curve with Anomaly Detection (z = {FIXED_Z:.5f})", fontsize=14)
        ax1.legend(ncol=2, fontsize=10)
        ax1.grid(True, alpha=0.3)

        sorted_idx = np.argsort(times_np)
        ax2.plot(times_np[sorted_idx], weighted_emax_np[sorted_idx], "k-", linewidth=2)
        ax2.fill_between(
            times_np[sorted_idx],
            0,
            weighted_emax_np[sorted_idx],
            alpha=0.3,
        )
        ax2.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label=f"Threshold = {threshold}")
        ax2.set_xlabel("MJD", fontsize=12)
        ax2.set_ylabel("Weighted Emax", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax1.set_xlim(ax2.get_xlim())

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "light_curve_anomaly.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: Could not create light curve plot - {exc}")


def print_summary_table(standard_samples, anomaly_samples):
    """Display summary statistics for both fits."""
    print("\n" + "=" * 60)
    print("Parameter Statistics")
    print("=" * 60)
    print(f"{'Parameter':<12} {'Standard':>20} {'Anomaly':>20}")
    print("-" * 60)
    for param in STANDARD_PARAM_NAMES:
        std_mean = standard_samples[param].mean()
        std_std = standard_samples[param].std()
        anom_mean = anomaly_samples[param].mean()
        anom_std = anomaly_samples[param].std()
        print(f"{param:<12} {std_mean:>10.4f} ± {std_std:<8.4f} {anom_mean:>10.4f} ± {anom_std:<8.4f}")

    if "log_p" in anomaly_samples.columns:
        logp_mean = anomaly_samples["log_p"].mean()
        logp_std = anomaly_samples["log_p"].std()
        print(f"{'log_p':<12} {'N/A':>20} {logp_mean:>10.4f} ± {logp_std:<8.4f}")


def main():
    print("Running anomaly detection example with fixed heliocentric redshift:")
    print(f"  Supernova ID: {SUPERNOVA_ID}")
    print(f"  z_hel = {FIXED_Z}")

    standard_run, standard_samples = run_nested_sampling(
        PRIOR_BOUNDS_STANDARD,
        STANDARD_PARAM_NAMES,
        loglikelihood_standard,
        label="standard",
        num_mcmc_steps=NUM_MCMC_STEPS_STANDARD,
    )

    anomaly_run, anomaly_samples = run_nested_sampling(
        PRIOR_BOUNDS_ANOMALY,
        ANOMALY_PARAM_NAMES,
        loglikelihood_anomaly,
        label="anomaly",
        num_mcmc_steps=NUM_MCMC_STEPS_ANOMALY,
    )

    weighted_emax = compute_weighted_emax(anomaly_run)
    emax_path = os.path.join(OUTPUT_DIR, "anomaly_weighted_emax.txt")
    np.savetxt(emax_path, weighted_emax)
    print(f"Saved weighted emax to {emax_path}")

    create_corner_plots(standard_samples, anomaly_samples)
    create_light_curve_plot(anomaly_samples, weighted_emax)
    print_summary_table(standard_samples, anomaly_samples)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Generated plots:")
    print("  - corner_comparison.png")
    print("  - corner_anomaly_logp.png")
    print("  - light_curve_anomaly.png")


if __name__ == "__main__":
    main()
