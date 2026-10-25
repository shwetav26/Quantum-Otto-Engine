"""
make_report_assets.py - regenerates every graph, CSV and key number used in the final report/PPT
with the SAME functions the app uses.  Run:  python make_report_assets.py
Outputs go to the results/ folder.
"""
import json
import os

import numpy as np

import plotting as pl
import quantum_engine as q
import simulation_manager as sm

OUT = "results"
os.makedirs(OUT, exist_ok=True)
P = dict(q.DEFAULT_PARAMS)
rho_ent, rho_ref = q.state_bell(), q.state_plus_plus()
ent, ref = q.simulate(P, rho_ent), q.simulate(P, rho_ref)
twin = q.simulate(P, sm.reference_state(sm.REF_TWIN, P))
df_p = sm.sweep_entanglement(P, 41)
df_g = sm.sweep_decoherence(P, np.linspace(0, 5, 41))
df_p.to_csv(f"{OUT}/entanglement_sweep.csv", index=False)
df_g.to_csv(f"{OUT}/decoherence_sweep.csv", index=False)

series = {"entangled (Bell)": sm.work_per_cycle(P, rho_ent), "product |+>|+>": sm.work_per_cycle(P, rho_ref),
          "dephased copy": sm.work_per_cycle(P, sm.reference_state(sm.REF_TWIN, P)),
          "product |00>": sm.work_per_cycle(P, q.state_product_00())}
grids = {}
for obj in ("work", "efficiency"):
    df_o, best = sm.grid_search(P, "wC", np.linspace(0.3, 1.9, 12), "TC", np.linspace(0.2, 3.0, 12), rho_ent, obj, "steady", 0.01)
    df_o.to_csv(f"{OUT}/optimization_{obj}.csv", index=False)
    grids[obj] = (df_o, best)

figs = {"cycle_energy": pl.fig_cycle_energy(ent, ref, sm.REF_PLUS), "heat_work": pl.fig_heat_work(ent, ref),
        "work_vs_p": pl.fig_work_vs_p(df_p), "eff_vs_p": pl.fig_eff_vs_p(df_p),
        "eff_vs_gamma": pl.fig_eff_vs_gamma(df_g, "plus"), "advantage_vs_gamma": pl.fig_advantage_vs_gamma(df_g),
        "work_per_cycle": pl.fig_work_per_cycle(series, ent["steady"]["W"]),
        "opt_work": pl.fig_opt_map(*grids["work"][0:1], "wC", "TC", "work", grids["work"][1]),
        "opt_eff": pl.fig_opt_map(*grids["efficiency"][0:1], "wC", "TC", "efficiency", grids["efficiency"][1])}
for name, fig in figs.items():
    fig.savefig(f"{OUT}/{name}.png", dpi=170)

# ---- reproducible stress test of the Carnot / Clausius bounds (random parameters, fixed seed)
rng = np.random.default_rng(2026)
stat = {"n": 0, "engines_g0": 0, "carnot_viol_g0": 0, "clausius_viol_g0": 0,
        "engines_noise": 0, "carnot_viol_noise": 0, "clausius_viol_noise": 0, "n_noise": 0, "n_g0": 0}
for _ in range(2000):
    TC = rng.uniform(0.2, 3); wC = rng.uniform(0.2, 3)
    pp = dict(TH=TC + rng.uniform(0.3, 8), TC=TC, wH=wC + rng.uniform(0.2, 3), wC=wC, J=rng.uniform(0, 2),
              kappa=10 ** rng.uniform(-1, 1.5), gamma=0.0)
    for tag, g in (("g0", 0.0), ("noise", 10 ** rng.uniform(-2, 1))):
        t = q.simulate(dict(pp, gamma=g), rho_ent)["steady"]
        stat["n_" + tag] += 1
        if t["is_engine"]:
            stat["engines_" + tag] += 1
            if t["eta"] > q.carnot_efficiency(pp["TH"], pp["TC"]) + 1e-9:
                stat["carnot_viol_" + tag] += 1
        if t["clausius"] < -1e-9:
            stat["clausius_viol_" + tag] += 1

def row(sim, key):
    c = sim[key]
    return {"QH": c["QH"], "QC": c["QC"], "W": c["W"], "eta": None if np.isnan(c["eta"]) else c["eta"],
            "dU": c["dU"], "eps": c["eps"], "U": c["U"], "conc": c["conc"], "status": c["status"]}

g = df_g
cross = {t: sm.find_zero_crossings(g["gamma"], g[f"dW_first_{t}"])[0] for t in ("plus", "p00", "twin")}
key = {
    "params": P, "carnot": q.carnot_efficiency(P["TH"], P["TC"]), "otto_uncoupled": q.otto_efficiency(P["wH"], P["wC"]),
    "bell": {"first": row(ent, "first"), "steady": row(ent, "steady"), "n_cycles": ent["n_cycles"]},
    "plusplus": {"first": row(ref, "first"), "steady": row(ref, "steady")},
    "twin": {"first": row(twin, "first"), "steady": row(twin, "steady")},
    "memory_hot_isochore_gamma_default": q.memory_after_hot_isochore(P, rho_ent, rho_ref),
    "memory_exp_minus_kappa_tau": float(np.exp(-P["kappa"] * q.TAU)),
    "dW_first": {t: [float(g["gamma"].iloc[i]) for i in (0,)] + [float(g[f"dW_first_{t}"].iloc[0]), float(g[f"dW_first_{t}"].iloc[-1])] for t in ("plus", "p00", "twin")},
    "crossings": cross,
    "gamma_sweep_W_steady": [float(g["W_steady_ent"].iloc[0]), float(g["W_steady_ent"].iloc[-1])],
    "gamma_sweep_eta_steady": [float(g["eta_steady_ent"].iloc[0]), float(g["eta_steady_ent"].iloc[-1])],
    "p_sweep": {"W_first_p0": float(df_p["W_first"].iloc[0]), "W_first_bell": float(df_p["W_first"].iloc[20]),
                "W_first_p1": float(df_p["W_first"].iloc[-1]),
                "p_where_W_first_zero": sm.find_zero_crossings(df_p["p"], df_p["W_first"])[0],
                "n_first_cycle_engine": int(df_p["engine_first"].sum()), "n_points": len(df_p)},
    "works_per_cycle": {k: [float(x) for x in v] for k, v in series.items()},
    "optimization": {o: {"best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in b.items()} if b else None,
                         "n_valid": int(d["valid"].sum()), "n_total": len(d)} for o, (d, b) in grids.items()},
    "stress_test": stat,
}
with open(f"{OUT}/key_numbers.json", "w") as f:
    json.dump(key, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print(json.dumps(key, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))[:6000])
