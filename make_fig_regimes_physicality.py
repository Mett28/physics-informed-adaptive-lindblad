from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils_io import load_npz

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_STD = SCRIPT_DIR / "outputs" / "regimes_standard"
BASE_PHY = SCRIPT_DIR / "outputs" / "regimes_physinf"
FIGDIR = SCRIPT_DIR / "figures"


def summarize_std(run):
    trace_mean = float(np.mean(np.asarray(run.arr["trace_err"])))
    lam_min = float(np.min(np.asarray(run.arr["lambda_min"])))
    return trace_mean, lam_min


def summarize_phy(run):
    trace_mean = float(np.mean(np.asarray(run.arr["trace_err"])))
    lam_min = float(np.min(np.asarray(run.arr["lambda_min"])))
    corr_count = int(np.asarray(run.arr["correction_count"]).item())
    corr_amp = np.asarray(run.arr["corr_amp"], dtype=float)
    corr_max = float(np.max(corr_amp))
    return trace_mean, lam_min, corr_count, corr_max


def main():
    regimes = ["regime1", "regime2", "regime3"]
    titles = [
        "Regime I: baseline",
        "Regime II: stronger dephasing / long time",
        "Regime III: stiffness-sensitive",
    ]

    std_trace = []
    phy_trace = []
    std_lam = []
    phy_lam = []
    phy_corr = []
    phy_corr_amp = []

    # Her rejimde tol=1e-3 dosyasını karşılaştırıyoruz
    for reg in regimes:
        std_path = BASE_STD / reg / "std_tol1e-03.npz"
        phy_path = BASE_PHY / reg / "phys_tol1e-03.npz"

        std = load_npz(std_path)
        phy = load_npz(phy_path)

        tr_s, lam_s = summarize_std(std)
        tr_p, lam_p, cc_p, camp_p = summarize_phy(phy)

        std_trace.append(tr_s)
        phy_trace.append(tr_p)
        std_lam.append(lam_s)
        phy_lam.append(lam_p)
        phy_corr.append(cc_p)
        phy_corr_amp.append(camp_p)

    x = np.arange(len(regimes))
    w = 0.34

    fig = plt.figure(figsize=(8.0, 8.8))
    gs = fig.add_gridspec(3, 1, hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Panel 1: mean trace error
    ax1.bar(x - w/2, std_trace, width=w, label="Standard adaptive RK4")
    ax1.bar(x + w/2, phy_trace, width=w, label="Physics-informed adaptive RK4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Regime I", "Regime II", "Regime III"])
    ax1.set_ylabel(r"Mean trace error")
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6)
    ax1.legend(frameon=False, loc="best")

    # Panel 2: minimum eigenvalue
    ax2.bar(x - w/2, std_lam, width=w, label="Standard adaptive RK4")
    ax2.bar(x + w/2, phy_lam, width=w, label="Physics-informed adaptive RK4")
    ax2.axhline(0.0, linestyle="--", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Regime I", "Regime II", "Regime III"])
    ax2.set_ylabel(r"Minimum eigenvalue")
    ax2.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6)

    # Panel 3: correction count
    bars = ax3.bar(x, phy_corr, width=0.5, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Regime I", "Regime II", "Regime III"])
    ax3.set_ylabel("Correction count")
    ax3.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6)

    for rect, val, amp in zip(bars, phy_corr, phy_corr_amp):
        ax3.annotate(
            f"{val}\nmax={amp:.1e}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_regimes_physicality.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()