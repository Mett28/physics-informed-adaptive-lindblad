from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils_io import load_npz

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PHY = SCRIPT_DIR / "outputs" / "regimes_physinf"
FIGDIR = SCRIPT_DIR / "figures"


def main():
    # En zorlayıcı rejimden başlayalım
    run_path = BASE_PHY / "regime3" / "phys_tol1e-03.npz"
    print("Looking for file:", run_path)

    if not run_path.exists():
        raise FileNotFoundError(f"Physics-informed regime file not found:\n{run_path}")

    run = load_npz(run_path)

    t = np.asarray(run.arr["t"])
    eps_F = np.asarray(run.arr["eps_F"], dtype=float)
    eps_tr = np.asarray(run.arr["eps_tr"], dtype=float)
    eps_lam = np.asarray(run.arr["eps_lam"], dtype=float)
    eps_obs = np.asarray(run.arr["eps_obs"], dtype=float)
    E_phys = np.asarray(run.arr["E_phys"], dtype=float)
    corr_applied = np.asarray(run.arr["corr_applied"], dtype=int)

    tau_F = float(run.meta.get("tau_F", run.meta.get("tol", 1.0)))
    tau_tr = float(run.meta.get("tau_tr", 1e-10))
    tau_lam = float(run.meta.get("tau_lam", 1e-8))
    tau_obs = float(run.meta.get("tau_obs", 5.0 * run.meta.get("tol", 1.0)))

    # normalize
    yF = eps_F / max(tau_F, 1e-30)
    yTr = eps_tr / max(tau_tr, 1e-30)
    yLam = eps_lam / max(tau_lam, 1e-30)
    yObs = eps_obs / max(tau_obs, 1e-30)

    fig = plt.figure(figsize=(7.6, 9.0))
    gs = fig.add_gridspec(5, 1, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)

    # Panel 1: channel decomposition
    ax1.plot(t, yF, linewidth=1.6, label=r"$\epsilon_F/\tau_F$")
    ax1.plot(t, yTr, linewidth=1.6, label=r"$\epsilon_{\mathrm{tr}}/\tau_{\mathrm{tr}}$")
    ax1.plot(t, yLam, linewidth=1.6, label=r"$\epsilon_{\lambda}/\tau_{\lambda}$")
    ax1.plot(t, yObs, linewidth=1.6, label=r"$\epsilon_{\mathrm{obs}}/\tau_{\mathrm{obs}}$")
    ax1.axhline(1.0, linestyle="--", linewidth=0.9)
    ax1.set_ylabel("Normalized\nchannels")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax1.legend(frameon=False, loc="best", fontsize=8)

    # Panel 2: total E_phys
    ax2.plot(t, E_phys, linewidth=1.8)
    ax2.axhline(1.0, linestyle="--", linewidth=0.9)
    ax2.set_ylabel(r"$\widehat{\mathcal{E}}_{\mathrm{phys}}$")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.6)

    # Panel 3: dominant channel index
    stacked = np.vstack([yF, yTr, yLam, yObs])
    dominant = np.argmax(stacked, axis=0)
    ax3.plot(t, dominant, drawstyle="steps-mid", linewidth=1.5)
    ax3.set_ylabel("Dominant\nchannel")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["F", "TR", "LAM", "OBS"])
    ax3.grid(True, which="both", linestyle=":", linewidth=0.6)

    # Panel 4: correction events
    ax4.plot(t, corr_applied, drawstyle="steps-mid", linewidth=1.5)
    ax4.set_ylabel("Correction")
    ax4.set_yticks([0, 1])
    ax4.grid(True, which="both", linestyle=":", linewidth=0.6)

    # Panel 5: max channel only
    max_channel = np.max(stacked, axis=0)
    ax5.plot(t, max_channel, linewidth=1.6)
    ax5.axhline(1.0, linestyle="--", linewidth=0.9)
    ax5.set_xlabel(r"$t$")
    ax5.set_ylabel("Max normalized\nchannel")
    ax5.set_yscale("log")
    ax5.grid(True, which="both", linestyle=":", linewidth=0.6)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_controller_decomposition_regime3.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()