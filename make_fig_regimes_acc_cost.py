from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils_io import load_npz

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_STD = SCRIPT_DIR / "outputs" / "regimes_standard"
BASE_PHY = SCRIPT_DIR / "outputs" / "regimes_physinf"
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


def plot_regime(ax, regime_name: str, title: str):
    std_files = sorted((BASE_STD / regime_name).glob("std_tol*.npz"))
    phy_files = sorted((BASE_PHY / regime_name).glob("phys_tol*.npz"))

    std_runs = [load_npz(f) for f in std_files]
    phy_runs = [load_npz(f) for f in phy_files]

    sx = np.array([get_scalar(r, "channel_apps") for r in std_runs], dtype=float)
    sy = np.array([get_scalar(r, "eps_obs_T") for r in std_runs], dtype=float)
    slabels = [f"tol={r.meta.get('tol', np.nan):g}" for r in std_runs]

    px = np.array([get_scalar(r, "channel_apps") for r in phy_runs], dtype=float)
    py = np.array([get_scalar(r, "eps_obs_T") for r in phy_runs], dtype=float)
    plabels = [f"tol={r.meta.get('tol', np.nan):g}" for r in phy_runs]

    os = np.argsort(sx)
    op = np.argsort(px)

    ax.plot(
        sx[os], sy[os],
        marker="s", linewidth=1.8, markersize=5,
        label="Standard adaptive RK4"
    )
    ax.plot(
        px[op], py[op],
        marker="^", linewidth=1.8, markersize=6,
        label="Physics-informed adaptive RK4"
    )

    for i in os:
        ax.annotate(
            slabels[i],
            (sx[i], sy[i]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=6,
        )

    for i in op:
        ax.annotate(
            plabels[i],
            (px[i], py[i]),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)


def main():
    fig = plt.figure(figsize=(7.6, 11.0))
    gs = fig.add_gridspec(3, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    plot_regime(ax1, "regime1", "Regime I: baseline")
    plot_regime(ax2, "regime2", "Regime II: stronger dephasing / long time")
    plot_regime(ax3, "regime3", "Regime III: stiffness-sensitive")

    ax1.set_ylabel(r"Final observable error, $\epsilon_{\mathrm{obs}}(T)$")
    ax2.set_ylabel(r"Final observable error, $\epsilon_{\mathrm{obs}}(T)$")
    ax3.set_ylabel(r"Final observable error, $\epsilon_{\mathrm{obs}}(T)$")
    ax3.set_xlabel("Channel applications")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_regimes_acc_cost.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()