from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils_io import load_npz

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_STD = SCRIPT_DIR / "outputs" / "regimes_standard_stress"
BASE_PHY = SCRIPT_DIR / "outputs" / "regimes_physinf_stress"
FIGDIR = SCRIPT_DIR / "figures"


def get_scalar(run, key, default=np.nan):
    if key in run.arr:
        try:
            return float(np.asarray(run.arr[key]).item())
        except Exception:
            return default
    if key in run.meta:
        try:
            return float(run.meta[key])
        except Exception:
            return default
    return default


def load_regime(regime):
    std_files = sorted((BASE_STD / regime).glob("std_tol*.npz"))
    phy_files = sorted((BASE_PHY / regime).glob("phys_tol*.npz"))

    std_runs = [load_npz(f) for f in std_files]
    phy_runs = [load_npz(f) for f in phy_files]

    return std_runs, phy_runs


def extract(runs):
    tol = np.array([get_scalar(r, "tol") for r in runs], dtype=float)
    acc = np.array([get_scalar(r, "accepted_steps") for r in runs], dtype=float)
    rej = np.array([get_scalar(r, "rejected_steps") for r in runs], dtype=float)
    apps = np.array([get_scalar(r, "channel_apps") for r in runs], dtype=float)
    err = np.array([get_scalar(r, "eps_obs_T") for r in runs], dtype=float)

    order = np.argsort(tol)[::-1]  # loose to strict
    return tol[order], acc[order], rej[order], apps[order], err[order]


def plot_regime(ax_err, ax_rej, regime, title):
    std_runs, phy_runs = load_regime(regime)

    tol_s, acc_s, rej_s, apps_s, err_s = extract(std_runs)
    tol_p, acc_p, rej_p, apps_p, err_p = extract(phy_runs)

    labels = [f"{t:g}" for t in tol_s]
    x = np.arange(len(labels))
    w = 0.36

    ax_err.plot(apps_s, err_s, marker="s", linewidth=1.8, label="Standard adaptive RK4")
    ax_err.plot(apps_p, err_p, marker="^", linewidth=1.8, label="Physics-informed adaptive RK4")
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_title(title)
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{obs}}(T)$")
    ax_err.grid(True, which="both", linestyle=":", linewidth=0.6)

    ax_rej.bar(x - w/2, rej_s, width=w, label="Standard adaptive RK4")
    ax_rej.bar(x + w/2, rej_p, width=w, label="Physics-informed adaptive RK4")
    ax_rej.set_xticks(x)
    ax_rej.set_xticklabels(labels)
    ax_rej.set_xlabel(r"Tolerance $\tau_F$")
    ax_rej.set_ylabel("Rejected steps")
    ax_rej.grid(True, axis="y", linestyle=":", linewidth=0.6)


def main():
    fig = plt.figure(figsize=(8.0, 8.4))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    plot_regime(ax11, ax12, "regime2", "Stress Regime II")
    plot_regime(ax21, ax22, "regime3", "Stress Regime III")

    handles, labels = ax11.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_stress_rejections.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()