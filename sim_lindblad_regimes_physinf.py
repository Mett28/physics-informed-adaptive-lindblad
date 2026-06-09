from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_STD = ROOT / "outputs" / "regimes_standard"
OUT_PHY = ROOT / "outputs" / "regimes_physinf"


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def op_on_site(op: np.ndarray, site: int, N: int) -> np.ndarray:
    I = np.eye(2, dtype=complex)
    ops = [I] * N
    ops[site] = op
    return kron_all(ops)


def two_site_op(op1: np.ndarray, i: int, op2: np.ndarray, j: int, N: int) -> np.ndarray:
    I = np.eye(2, dtype=complex)
    ops = [I] * N
    ops[i] = op1
    ops[j] = op2
    return kron_all(ops)


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    drho = -1j * (H @ rho - rho @ H)
    for L in Ls:
        LdL = L.conj().T @ L
        drho += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return drho


def rk4_step(rho: np.ndarray, dt: float, H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    k1 = lindblad_rhs(rho, H, Ls)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, Ls)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, Ls)
    k4 = lindblad_rhs(rho + dt * k3, H, Ls)
    rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    tr = np.trace(rho_new)
    if abs(tr) > 0:
        rho_new = rho_new / tr
    return rho_new


def rk4_two_half_steps(rho: np.ndarray, dt: float, H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    return rk4_step(rk4_step(rho, 0.5 * dt, H, Ls), 0.5 * dt, H, Ls)


def magnetizations(rho: np.ndarray, sz_ops: list[np.ndarray]) -> np.ndarray:
    return np.array([np.real(np.trace(rho @ op)) for op in sz_ops], dtype=float)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def make_meta(**kwargs) -> np.ndarray:
    return np.array(json.dumps(kwargs))


def hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def min_eig(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(hermitize(rho))
    return float(np.min(np.real(vals)))


def positivity_projection(rho: np.ndarray):
    rho_h = hermitize(rho)
    vals, vecs = np.linalg.eigh(rho_h)
    neg_mass = float(np.sum(np.abs(vals[vals < 0.0])))
    vals_clip = np.clip(vals, 0.0, None)
    rho_proj = vecs @ np.diag(vals_clip) @ vecs.conj().T
    tr = np.trace(rho_proj)
    if abs(tr) > 0:
        rho_proj = rho_proj / tr
    amp = float(np.linalg.norm(rho_proj - rho_h, ord="fro"))
    return rho_proj, amp, neg_mass


def build_model(gamma_phi=0.025, h=0.35, N=6, Jxy=1.0, Delta=1.2):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        H += Jxy * two_site_op(sx, i, sx, i + 1, N)
        H += Jxy * two_site_op(sy, i, sy, i + 1, N)
        H += Delta * two_site_op(sz, i, sz, i + 1, N)

    for i in range(N):
        H += h * op_on_site(sz, i, N)

    Ls = [np.sqrt(gamma_phi) * op_on_site(sz, i, N) for i in range(N)]

    up = np.array([[1], [0]], dtype=complex)
    dn = np.array([[0], [1]], dtype=complex)

    psi0 = up
    for state in [up, up, dn, dn, dn]:
        psi0 = np.kron(psi0, state)
    rho0 = psi0 @ psi0.conj().T
    sz_ops = [op_on_site(sz, i, N) for i in range(N)]

    return {
        "N": N,
        "H": H,
        "Ls": Ls,
        "rho0": rho0,
        "sz_ops": sz_ops,
        "params": {
            "Jxy": Jxy,
            "Delta": Delta,
            "h": h,
            "gamma_phi": gamma_phi,
        },
    }


def observable_error_against_ref(t_num, mi_num, t_ref, mi_ref):
    errs = []
    for k, tk in enumerate(t_num):
        j = int(np.argmin(np.abs(t_ref - tk)))
        errs.append(np.max(np.abs(mi_num[k] - mi_ref[j])))
    return np.array(errs, dtype=float)


def run_reference(model, T=10.0, dt=0.0015):
    H, Ls, rho0, sz_ops = model["H"], model["Ls"], model["rho0"], model["sz_ops"]
    t = np.arange(0.0, T + 1e-15, dt)
    rho = rho0.copy()
    mi = np.zeros((len(t), len(sz_ops)), dtype=float)

    for k in range(len(t)):
        mi[k, :] = magnetizations(rho, sz_ops)
        if k < len(t) - 1:
            rho = rk4_step(rho, dt, H, Ls)

    return {"t": t, "mi": mi, "rho_final": rho}


def run_standard_adaptive(model, ref, tol=1e-3, T=10.0, dt0=0.05, dt_min=0.0015, dt_max=0.08):
    H, Ls, rho0, sz_ops, N = model["H"], model["Ls"], model["rho0"], model["sz_ops"], model["N"]

    t_vals = [0.0]
    rho = rho0.copy()
    mi_list = [magnetizations(rho, sz_ops)]
    trace_list = [float(abs(np.trace(rho) - 1.0))]
    lam_list = [min_eig(rho)]
    dt_list = [dt0]
    eps_list = [0.0]
    accepted_marks = []

    accepted_steps = 0
    rejected_steps = 0
    dt = dt0

    while t_vals[-1] < T - 1e-14:
        t_now = t_vals[-1]
        if t_now + dt > T:
            dt = T - t_now

        rho_big = rk4_step(rho, dt, H, Ls)
        rho_half = rk4_two_half_steps(rho, dt, H, Ls)
        eps = float(np.linalg.norm(rho_half - rho_big, ord="fro"))

        if eps <= tol or dt <= dt_min * 1.0001:
            rho = rho_half
            t_vals.append(t_now + dt)
            mi_list.append(magnetizations(rho, sz_ops))
            trace_list.append(float(abs(np.trace(rho) - 1.0)))
            lam_list.append(min_eig(rho))
            dt_list.append(dt)
            eps_list.append(eps)
            accepted_marks.append(1)
            accepted_steps += 1

            factor = 1.6 if eps < 1e-16 else min(1.7, max(0.4, 0.9 * (tol / eps) ** 0.2))
            dt = min(dt_max, max(dt_min, dt * factor))
        else:
            accepted_marks.append(0)
            rejected_steps += 1
            dt = max(dt_min, 0.45 * dt)

    t = np.asarray(t_vals)
    mi = np.asarray(mi_list)
    eps_obs = observable_error_against_ref(t, mi, ref["t"], ref["mi"])

    return {
        "t": t,
        "dt": np.asarray(dt_list),
        "trace_err": np.asarray(trace_list),
        "lambda_min": np.asarray(lam_list),
        "mi": mi,
        "eps": np.asarray(eps_list),
        "accepted": np.asarray(accepted_marks, dtype=int),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "channel_apps": 3 * (accepted_steps + rejected_steps),
        "eps_obs_T": float(eps_obs[-1]),
        "eps_true_T": float(np.max(eps_obs)),
    }


def run_physinf_adaptive(
    model,
    ref,
    tol=1e-3,
    T=10.0,
    dt0=0.05,
    dt_min=0.0015,
    dt_max=0.08,
    tau_tr=1e-10,
    tau_lam=1e-8,
    tau_obs_factor=5.0,
    w_F=1.0,
    w_tr=1.0,
    w_lam=1.0,
    w_obs=1.0,
):
    H, Ls, rho0, sz_ops, N = model["H"], model["Ls"], model["rho0"], model["sz_ops"], model["N"]

    tau_F = tol
    tau_obs = tau_obs_factor * tol

    t_vals = [0.0]
    rho = rho0.copy()

    mi_list = [magnetizations(rho, sz_ops)]
    trace_list = [float(abs(np.trace(rho) - 1.0))]
    lam_list = [min_eig(rho)]
    dt_list = [dt0]

    eps_F_list = [0.0]
    eps_tr_list = [0.0]
    eps_lam_list = [0.0]
    eps_obs_list = [0.0]
    E_phys_list = [0.0]

    corr_applied = [0]
    corr_amp = [0.0]
    corr_neg_mass = [0.0]
    accepted_marks = []

    accepted_steps = 0
    rejected_steps = 0
    correction_count = 0
    dt = dt0

    while t_vals[-1] < T - 1e-14:
        t_now = t_vals[-1]
        if t_now + dt > T:
            dt = T - t_now

        rho_big = rk4_step(rho, dt, H, Ls)
        rho_half = rk4_two_half_steps(rho, dt, H, Ls)
        rho_trial = rho_half

        eps_F = float(np.linalg.norm(rho_half - rho_big, ord="fro"))
        eps_tr = float(abs(np.trace(rho_trial) - 1.0))
        lam = min_eig(rho_trial)
        eps_lam = float(max(0.0, -lam))

        m_big = magnetizations(rho_big, sz_ops)
        m_half = magnetizations(rho_half, sz_ops)
        eps_obs = float(np.max(np.abs(m_half - m_big)))

        E_phys = (
            w_F * (eps_F / max(tau_F, 1e-30))
            + w_tr * (eps_tr / max(tau_tr, 1e-30))
            + w_lam * (eps_lam / max(tau_lam, 1e-30))
            + w_obs * (eps_obs / max(tau_obs, 1e-30))
        )

        accept = (E_phys <= 1.0) and (lam >= -tau_lam)

        used_corr = 0
        used_amp = 0.0
        used_neg_mass = 0.0

        if not accept and eps_lam > 0.0:
            rho_corr, amp, neg_mass = positivity_projection(rho_trial)

            eps_tr_corr = float(abs(np.trace(rho_corr) - 1.0))
            lam_corr = min_eig(rho_corr)
            eps_lam_corr = float(max(0.0, -lam_corr))
            m_corr = magnetizations(rho_corr, sz_ops)
            eps_obs_corr = float(np.max(np.abs(m_corr - m_big)))
            eps_F_corr = float(np.linalg.norm(rho_corr - rho_big, ord="fro"))

            E_phys_corr = (
                w_F * (eps_F_corr / max(tau_F, 1e-30))
                + w_tr * (eps_tr_corr / max(tau_tr, 1e-30))
                + w_lam * (eps_lam_corr / max(tau_lam, 1e-30))
                + w_obs * (eps_obs_corr / max(tau_obs, 1e-30))
            )

            if (E_phys_corr <= 1.0) and (lam_corr >= -tau_lam):
                rho_trial = rho_corr
                eps_F = eps_F_corr
                eps_tr = eps_tr_corr
                eps_lam = eps_lam_corr
                eps_obs = eps_obs_corr
                E_phys = E_phys_corr
                lam = lam_corr
                used_corr = 1
                used_amp = amp
                used_neg_mass = neg_mass
                correction_count += 1
                accept = True

        if accept or dt <= dt_min * 1.0001:
            rho = rho_trial
            t_vals.append(t_now + dt)
            mi_list.append(magnetizations(rho, sz_ops))
            trace_list.append(float(abs(np.trace(rho) - 1.0)))
            lam_list.append(min_eig(rho))
            dt_list.append(dt)

            eps_F_list.append(eps_F)
            eps_tr_list.append(eps_tr)
            eps_lam_list.append(eps_lam)
            eps_obs_list.append(eps_obs)
            E_phys_list.append(E_phys)

            corr_applied.append(used_corr)
            corr_amp.append(used_amp)
            corr_neg_mass.append(used_neg_mass)
            accepted_marks.append(1)
            accepted_steps += 1

            shrink_measure = max(E_phys, 1e-16)
            factor = min(1.7, max(0.4, 0.9 * (1.0 / shrink_measure) ** 0.2))
            dt = min(dt_max, max(dt_min, dt * factor))
        else:
            accepted_marks.append(0)
            rejected_steps += 1
            dt = max(dt_min, 0.45 * dt)

    t = np.asarray(t_vals)
    mi = np.asarray(mi_list)
    eps_obs_ref = observable_error_against_ref(t, mi, ref["t"], ref["mi"])

    return {
        "t": t,
        "dt": np.asarray(dt_list),
        "trace_err": np.asarray(trace_list),
        "lambda_min": np.asarray(lam_list),
        "mi": mi,
        "eps_F": np.asarray(eps_F_list),
        "eps_tr": np.asarray(eps_tr_list),
        "eps_lam": np.asarray(eps_lam_list),
        "eps_obs": np.asarray(eps_obs_list),
        "E_phys": np.asarray(E_phys_list),
        "corr_applied": np.asarray(corr_applied, dtype=int),
        "corr_amp": np.asarray(corr_amp),
        "corr_neg_mass": np.asarray(corr_neg_mass),
        "accepted": np.asarray(accepted_marks, dtype=int),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "correction_count": correction_count,
        "channel_apps": 3 * (accepted_steps + rejected_steps),
        "eps_obs_T": float(eps_obs_ref[-1]),
        "eps_true_T": float(np.max(eps_obs_ref)),
    }


def save_run(path: Path, data: dict, meta: dict):
    save_npz(path, **data, meta_json=make_meta(**meta))


def run_regime(name: str, gamma_phi: float, h: float, T: float):
    print(f"Running {name}: gamma_phi={gamma_phi}, h={h}, T={T}")
    model = build_model(gamma_phi=gamma_phi, h=h)
    ref = run_reference(model, T=T, dt=0.0015)

    for tol in [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]:
        std = run_standard_adaptive(model, ref, tol=tol, T=T)
        std_meta = {
            "scheme": "standard_adaptive",
            "regime": name,
            "tol": tol,
            "gamma_phi": gamma_phi,
            "h": h,
            "T": T,
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

        phy = run_physinf_adaptive(model, ref, tol=tol, T=T)
        phy_meta = {
            "scheme": "physics_informed_adaptive",
            "regime": name,
            "tol": tol,
            "gamma_phi": gamma_phi,
            "h": h,
            "T": T,
            "accepted_steps": phy["accepted_steps"],
            "rejected_steps": phy["rejected_steps"],
            "channel_apps": phy["channel_apps"],
            "correction_count": phy["correction_count"],
            "eps_obs_T": phy["eps_obs_T"],
            "eps_true_T": phy["eps_true_T"],
        }
        save_run(
            OUT_PHY / name / f"phys_tol{tol:.0e}.npz".replace("+0", ""),
            phy,
            phy_meta,
        )


def main():
    run_regime("regime1", gamma_phi=0.025, h=0.35, T=10.0)
    run_regime("regime2", gamma_phi=0.15, h=0.35, T=20.0)
    run_regime("regime3", gamma_phi=0.5, h=1.5, T=10.0)
    print("All regimes completed.")


if __name__ == "__main__":
    main()