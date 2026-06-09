from pathlib import Path
import numpy as np
from utils_io import load_npz

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_STD = SCRIPT_DIR / "outputs" / "regimes_standard_stress"
BASE_PHY = SCRIPT_DIR / "outputs" / "regimes_physinf_stress"
TABDIR = SCRIPT_DIR / "tables"


def get_scalar(run, key, default=None):
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


def fmt(x):
    if x is None:
        return "--"
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.6g}"


def max_ephys(run):
    if "E_phys" in run.arr:
        return float(np.max(np.asarray(run.arr["E_phys"], dtype=float)))
    return None


def corr_count(run):
    if "correction_count" in run.arr:
        return int(np.asarray(run.arr["correction_count"]).item())
    return 0


def main():
    regimes = ["regime2", "regime3"]
    rows = []

    for reg in regimes:
        for f in sorted((BASE_STD / reg).glob("std_tol*.npz")):
            run = load_npz(f)
            rows.append([
                reg,
                "Std. adapt. RK4",
                get_scalar(run, "tol", run.meta.get("tol", None)),
                get_scalar(run, "accepted_steps"),
                get_scalar(run, "rejected_steps"),
                0,
                get_scalar(run, "channel_apps"),
                get_scalar(run, "eps_obs_T"),
                max_ephys(run),
            ])

        for f in sorted((BASE_PHY / reg).glob("phys_tol*.npz")):
            run = load_npz(f)
            rows.append([
                reg,
                "Phys.-inf. RK4",
                get_scalar(run, "tol", run.meta.get("tol", None)),
                get_scalar(run, "accepted_steps"),
                get_scalar(run, "rejected_steps"),
                corr_count(run),
                get_scalar(run, "channel_apps"),
                get_scalar(run, "eps_obs_T"),
                max_ephys(run),
            ])

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "table_stress_summary.tex"

    lines = []
    lines.append(r"\begin{tabular}{@{}llrrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Regime & Scheme & tol & acc. & rej. & corr. & apps & $\epsilon_{\mathrm{obs}}(T)$ & max $\widehat{\mathcal E}_{\mathrm{phys}}$ \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        lines.append(" & ".join([
            str(row[0]),
            str(row[1]),
            fmt(row[2]),
            fmt(row[3]),
            fmt(row[4]),
            fmt(row[5]),
            fmt(row[6]),
            fmt(row[7]),
            fmt(row[8]),
        ]) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()