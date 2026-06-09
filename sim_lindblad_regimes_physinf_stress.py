from pathlib import Path
from sim_lindblad_regimes_physinf import (
    build_model,
    run_reference,
    run_standard_adaptive,
    run_physinf_adaptive,
    save_npz,
    make_meta,
)

ROOT = Path(__file__).resolve().parent

OUT_STD = ROOT / "outputs" / "regimes_standard_stress"
OUT_PHY = ROOT / "outputs" / "regimes_physinf_stress"


def save_run(path: Path, data: dict, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, **data, meta_json=make_meta(**meta))


def run_regime_stress(name: str, gamma_phi: float, h: float, T: float):
    print(f"Running STRESS {name}: gamma_phi={gamma_phi}, h={h}, T={T}")

    model = build_model(gamma_phi=gamma_phi, h=h)
    ref = run_reference(model, T=T, dt=0.0015)

    tol_values = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]

    for tol in tol_values:
        print(f"  tol = {tol:g}")

        std = run_standard_adaptive(
            model,
            ref,
            tol=tol,
            T=T,
            dt0=0.20,
            dt_min=0.0005,
            dt_max=0.20,
        )

        std_meta = {
            "scheme": "standard_adaptive",
            "variant": "stress",
            "regime": name,
            "tol": tol,
            "gamma_phi": gamma_phi,
            "h": h,
            "T": T,
            "dt0": 0.20,
            "dt_max": 0.20,
            "accepted_steps": std["accepted_steps"],
            "rejected_steps": std["rejected_steps"],
            "channel_apps": std["channel_apps"],
            "eps_obs_T": std["eps_obs_T"],
            "eps_true_T": std["eps_true_T"],
        }

        save_run(
            OUT_STD / name / f"std_tol{tol:.0e}.npz".replace("+0", ""),
            std,
            std_meta,
        )

        phy = run_physinf_adaptive(
            model,
            ref,
            tol=tol,
            T=T,
            dt0=0.20,
            dt_min=0.0005,
            dt_max=0.20,
            tau_tr=1e-12,
            tau_lam=1e-12,
            tau_obs_factor=0.5,
            w_F=1.0,
            w_tr=1.0,
            w_lam=5.0,
            w_obs=3.0,
        )

        phy_meta = {
            "scheme": "physics_informed_adaptive",
            "variant": "stress",
            "regime": name,
            "tol": tol,
            "gamma_phi": gamma_phi,
            "h": h,
            "T": T,
            "dt0": 0.20,
            "dt_max": 0.20,
            "tau_tr": 1e-12,
            "tau_lam": 1e-12,
            "tau_obs_factor": 0.5,
            "w_F": 1.0,
            "w_tr": 1.0,
            "w_lam": 5.0,
            "w_obs": 3.0,
            "accepted_steps": phy["accepted_steps"],
            "rejected_steps": phy["rejected_steps"],
            "correction_count": phy["correction_count"],
            "channel_apps": phy["channel_apps"],
            "eps_obs_T": phy["eps_obs_T"],
            "eps_true_T": phy["eps_true_T"],
        }

        save_run(
            OUT_PHY / name / f"phys_tol{tol:.0e}.npz".replace("+0", ""),
            phy,
            phy_meta,
        )


def main():
    run_regime_stress("regime2", gamma_phi=0.15, h=0.35, T=20.0)
    run_regime_stress("regime3", gamma_phi=0.5, h=1.5, T=10.0)
    print("Stress-test regime runs completed.")


if __name__ == "__main__":
    main()