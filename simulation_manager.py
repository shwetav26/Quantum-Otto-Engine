"""
simulation_manager.py
=====================
Parameter sweeps, the entangled-vs-reference comparison, a simple grid-search
optimiser and the automatic text explanation.  Uses quantum_engine.py only.
"""
import numpy as np
import pandas as pd
import quantum_engine as q

# Names of the three reference states offered in the comparison
REF_PLUS = "Product state |+>|+> (same energy as the Bell state)"
REF_00 = "Product state |00>"
REF_TWIN = "Dephased copy of the entangled state (control for coherence)"
REFERENCES = [REF_PLUS, REF_00, REF_TWIN]

PARAM_LABELS = {
    "TH": "TH (hot-bath temperature)", "TC": "TC (cold-bath temperature)",
    "wH": "wH (hot energy gap)", "wC": "wC (cold energy gap)",
    "J": "J (qubit-qubit interaction)", "kappa": "kappa (thermal contact rate)",
    "gamma": "gamma (decoherence rate)",
}


def reference_state(kind, p, p_ent=0.5):
    """Build the reference density matrix for the chosen comparison."""
    if kind == REF_PLUS:
        return q.state_plus_plus()
    if kind == REF_00:
        return q.state_product_00()
    if kind == REF_TWIN:
        return q.dephased_copy(q.state_p(p_ent), p["wH"], p["J"])
    raise ValueError(f"Unknown reference state: {kind}")


# ----------------------------------------------------------------------------
# One operating point -> one table row
# ----------------------------------------------------------------------------
def evaluate(p, rho0, label=""):
    """Run the engine for one parameter set and one initial state; return a flat dict."""
    s = q.simulate(p, rho0)
    f, t = s["first"], s["steady"]
    carnot = q.carnot_efficiency(p["TH"], p["TC"])
    eta_s = t["eta"]
    return {
        "state": label,
        "conc_initial": f["conc"][0], "U_initial": f["U"][0],
        "QH_first": f["QH"], "QC_first": f["QC"], "W_first": f["W"], "eta_first": f["eta"],
        "dU_first": f["dU"], "engine_first": f["is_engine"],
        "QH_steady": t["QH"], "QC_steady": t["QC"], "W_steady": t["W"], "eta_steady": eta_s,
        "engine_steady": t["is_engine"],
        "steady_above_carnot": bool(t["is_engine"] and eta_s > carnot + 1e-9),
        "clausius_steady": t["clausius"],
        "carnot": carnot, "otto": q.otto_efficiency(p["wH"], p["wC"]),
        "n_cycles": s["n_cycles"], "converged": s["converged"],
        "max_energy_error": s["max_energy_error"],
    }


def first_cycle_physical(df, suffix=""):
    """Mask of rows where the FIRST-cycle efficiency is both defined and not above Carnot.
    (The first cycle spends the starting energy, so eta can exceed Carnot; those points
    are excluded from efficiency plots and counted, never clipped.)"""
    eta = df["eta_first" + suffix]
    return eta.notna() & (eta <= df["carnot"] + 1e-9)


# ----------------------------------------------------------------------------
# Sweep 1: entanglement parameter p
# ----------------------------------------------------------------------------
def sweep_entanglement(p, n_points=41):
    """Vary p in |psi(p)> = sqrt(1-p)|00> + sqrt(p)|11> with all other inputs fixed."""
    rows = []
    for pv in np.linspace(0.0, 1.0, n_points):
        row = evaluate(p, q.state_p(pv), label=f"psi(p={pv:.3f})")
        row["p"] = pv
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sweep 2: decoherence rate gamma, entangled vs each reference
# ----------------------------------------------------------------------------
def sweep_decoherence(p, gammas, p_ent=0.5):
    """For every gamma run the entangled state and all reference states.
    Columns ending _ent belong to the entangled state; _ref_<k> to reference k."""
    gammas = np.asarray(gammas, dtype=float)
    short = {REF_PLUS: "plus", REF_00: "p00", REF_TWIN: "twin"}
    rows = []
    for g in gammas:
        pg = dict(p, gamma=float(g))
        ent = evaluate(pg, q.state_p(p_ent), "entangled")
        row = {"gamma": g}
        for key in ("conc_initial", "U_initial", "W_first", "eta_first", "QH_first", "W_steady", "eta_steady",
                    "steady_above_carnot", "carnot", "otto"):
            row[key + "_ent"] = ent[key]
        for kind in REFERENCES:
            ref = evaluate(pg, reference_state(kind, pg, p_ent), kind)
            tag = short[kind]
            for key in ("conc_initial", "U_initial", "W_first", "eta_first", "QH_first", "W_steady", "eta_steady"):
                row[f"{key}_{tag}"] = ref[key]
            row[f"dW_first_{tag}"] = ent["W_first"] - ref["W_first"]
            row[f"dW_steady_{tag}"] = ent["W_steady"] - ref["W_steady"]
            both = np.isfinite(ent["eta_first"]) and np.isfinite(ref["eta_first"])
            row[f"deta_first_{tag}"] = ent["eta_first"] - ref["eta_first"] if both else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def find_zero_crossings(x, y, tol=1e-12):
    """Where does y change sign?  Linear interpolation between the two neighbouring tested points.
    Returns (list_of_estimates, note).  The estimates are interpolated, NOT simulated values.
    Points where |y| < tol count as 'zero' and are not used as a crossing: a difference that is
    zero at the start and never changes sign afterwards has no threshold."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return [], "Not enough valid points."
    if np.all(np.abs(y) < tol):
        return [], "The difference is zero (within tolerance) at every tested point."
    nz = np.abs(y) >= tol                      # keep only clearly non-zero points
    xs, ys = x[nz], y[nz]
    found = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            found.append(float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])))
    if not found:
        return [], "No sign change was found inside the tested range."
    return found, "Estimated by linear interpolation between tested points."


# ----------------------------------------------------------------------------
# Memory of the initial state: work of each cycle
# ----------------------------------------------------------------------------
def work_per_cycle(p, rho0, n_cycles=12):
    """List of Wout for cycles 1..n (shows how fast the start state is forgotten)."""
    return q.simulate(p, rho0, n_forced=n_cycles)["works"]


# ----------------------------------------------------------------------------
# Optimisation by grid search
# ----------------------------------------------------------------------------
def grid_search(p, x_name, x_values, y_name, y_values, rho0, objective="work",
                cycle_kind="steady", w_min=0.01, progress=None):
    """Try every (x, y) pair on a grid.  Returns (DataFrame of all points, best row or None).
    objective   : 'work' (largest Wout) or 'efficiency' (largest eta with Wout >= w_min)
    cycle_kind  : 'steady' (engine after many cycles) or 'first' (starts from rho0)
    A point is VALID only if: parameters are physical, the engine takes heat (QH>0) and
    makes work (W>0), and (steady cycle only) eta does not exceed Carnot.
    This is the best point INSIDE THE TESTED GRID, not a universal optimum."""
    if x_name == y_name:
        raise ValueError("Choose two different parameters for the grid.")
    rows, total, done = [], len(x_values) * len(y_values), 0
    for xv in x_values:
        for yv in y_values:
            pt = dict(p)
            pt[x_name], pt[y_name] = float(xv), float(yv)
            row = {x_name: float(xv), y_name: float(yv), "valid": False, "reason": "",
                   "W": np.nan, "eta": np.nan, "QH": np.nan, "QC": np.nan}
            problems = q.check_params(pt)
            if problems:
                row["reason"] = "invalid parameters: " + " ".join(problems)
            else:
                s = q.simulate(pt, rho0)
                c = s["steady"] if cycle_kind == "steady" else s["first"]
                row.update(W=c["W"], eta=c["eta"], QH=c["QH"], QC=c["QC"])
                carnot = q.carnot_efficiency(pt["TH"], pt["TC"])
                if not c["is_engine"]:
                    row["reason"] = "not a heat engine (needs QH>0 and W>0)"
                elif cycle_kind == "steady" and not s["converged"]:
                    row["reason"] = "cycle did not converge"
                elif cycle_kind == "steady" and c["eta"] > carnot + 1e-9:
                    row["reason"] = "efficiency above Carnot (noise acts as extra energy source)"
                elif objective == "efficiency" and c["W"] < w_min:
                    row["reason"] = f"work below the minimum {w_min}"
                else:
                    row["valid"] = True
            rows.append(row)
            done += 1
            if progress is not None:
                progress(done / total)
    df = pd.DataFrame(rows)
    valid = df[df["valid"]]
    if valid.empty:
        return df, None
    key = "W" if objective == "work" else "eta"
    best = valid.loc[valid[key].idxmax()].to_dict()
    return df, best


# ----------------------------------------------------------------------------
# Automatic explanation written from the numbers
# ----------------------------------------------------------------------------
def explain_results(p, ent, ref, ref_name, df_g, memory):
    """Return a list of sentences built ONLY from the computed data (no fixed conclusions)."""
    out = []
    f_e, f_r = ent["first"], ref["first"]
    s_e, s_r = ent["steady"], ref["steady"]
    dW = f_e["W"] - f_r["W"]
    word = "more" if dW > 0 else "less"
    out.append(f"In cycle 1 the entangled state gives Wout = {f_e['W']:.4f}, the reference gives {f_r['W']:.4f}; "
               f"the entangled state extracts {abs(dW):.4f} {word} work (dW = {dW:+.4f}).")
    dU = f_e["U"][0] - f_r["U"][0]
    if abs(dU) > 1e-9:
        out.append(f"The two starting states do NOT have the same energy (difference {dU:+.4f}), so part of this "
                   f"difference comes from stored energy, not from entanglement.")
    else:
        out.append("Both starting states have the same energy, so the difference comes from how that energy is "
                   "distributed over the energy levels and from coherence.")
    out.append(f"Fraction of the initial-state difference that survives the hot isochore: {memory:.3f}. "
               + ("The hot bath erases most of it." if memory < 0.2 else "A noticeable part survives."))
    d_steady = abs(s_e["W"] - s_r["W"])
    if d_steady < 1e-8:
        out.append(f"After the engine settles (about {ent['n_cycles']} cycles) both states give the same work "
                   f"{s_e['W']:.4f}: the baths erase the initial state, so any advantage belongs to the early cycles only.")
    else:
        out.append(f"The settled cycles differ by {d_steady:.2e}; check convergence.")
    if s_e["is_engine"]:
        out.append(f"The settled engine has efficiency {s_e['eta']:.4f}; the uncoupled-qubit Otto value 1-wC/wH is "
                   f"{q.otto_efficiency(p['wH'], p['wC']):.4f} and the Carnot limit is "
                   f"{q.carnot_efficiency(p['TH'], p['TC']):.4f}.")
        if s_e["eta"] > q.carnot_efficiency(p["TH"], p["TC"]) + 1e-9:
            out.append("WARNING: the settled efficiency is above the Carnot limit. The decoherence noise is acting as "
                       "an extra energy source, so the two-bath Carnot bound does not apply to this operating point.")
    else:
        out.append("The settled cycle is not working as a heat engine at these inputs (" + s_e["status"] + ").")
    if df_g is not None and len(df_g) > 1:
        tag = {REF_PLUS: "plus", REF_00: "p00", REF_TWIN: "twin"}[ref_name]
        col = f"dW_first_{tag}"
        g, d = df_g["gamma"].values, df_g[col].values
        a0, a1 = abs(d[0]), abs(d[-1])
        trend = "shrinks" if a1 < a0 else "grows"
        out.append(f"Across the tested decoherence range (gamma = {g[0]:g} to {g[-1]:g}) the cycle-1 work difference "
                   f"{trend}: |dW| = {a0:.4f} at gamma = {g[0]:g} and {a1:.4f} at gamma = {g[-1]:g}.")
        cross, note = find_zero_crossings(g, d)
        if cross:
            out.append("dW changes sign near gamma = " + ", ".join(f"{c:.3f}" for c in cross)
                       + " (interpolated estimate, not a simulated value).")
        else:
            out.append("No decoherence threshold (sign change of dW) was found in the tested range. " + note)
        if ref_name == REF_TWIN:
            out.append("Note: the dephased copy is not guaranteed to be a product state, so a difference relative to it "
                       "measures the effect of coherence between energy levels, not of entanglement alone.")
    return out
