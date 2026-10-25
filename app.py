"""
app.py - Streamlit interface for the two-qubit quantum Otto engine simulator.
Run with:   streamlit run app.py
All physics is in quantum_engine.py; this file only collects inputs, calls the
functions and shows the results.
"""
import numpy as np
import pandas as pd
import streamlit as st

import plotting as pl
import quantum_engine as q
import simulation_manager as sm

st.set_page_config(page_title="Quantum Otto Engine Simulator", layout="wide")

# =============================================================================
# SECTION 1: Project overview
# =============================================================================
st.title("Work Extraction from Entangled Systems")
st.subheader("Efficiency and performance of a two-qubit quantum Otto engine under decoherence")
st.markdown(
    "**Research question:** How do entanglement and environmental decoherence affect the work extraction and "
    "efficiency of a two-qubit quantum Otto engine, and under what conditions, if any, does an entangled initial "
    "state provide an advantage over a non-entangled reference state?"
)
st.markdown(
    "This simulator takes two qubits as the working medium, runs the four stages of an Otto cycle on their "
    "4x4 density matrix, and calculates the heat, work and efficiency. It compares an entangled starting state "
    "with a non-entangled one, adds noise (decoherence) and searches a grid of settings for the best operating point."
)
with st.expander("The four stages of the quantum Otto cycle", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Hot isochore**  \nEnergy gap stays at wH. Qubits exchange heat QH with the hot bath.")
    c2.markdown("**2. Adiabatic expansion**  \nGap changes wH -> wC. No heat. Energy change = work done by the qubits.")
    c3.markdown("**3. Cold isochore**  \nGap stays at wC. Qubits exchange heat QC with the cold bath.")
    c4.markdown("**4. Adiabatic compression**  \nGap changes wC -> wH. No heat. Cycle is ready to repeat.")
    st.latex(r"H(\omega) = -\tfrac{\omega}{2}\,(\sigma_z\otimes I + I\otimes\sigma_z) - J\,\sigma_x\otimes\sigma_x")
    st.latex(r"\dot\rho = -i[H,\rho] + \kappa(\rho_{th}-\rho) + \gamma\sum_{j=1,2}(\sigma_z^{(j)}\rho\,\sigma_z^{(j)}-\rho)")
    st.caption("Units: hbar = kB = 1. Energies, gaps, temperatures and rates share one energy unit; time is in 1/energy. "
               "Sign convention: Q > 0 means heat flows into the qubits; W > 0 means work done by the qubits.")

# =============================================================================
# SECTION 2: Simulation inputs (sidebar)
# =============================================================================
sb = st.sidebar
sb.header("Simulation inputs")
sb.caption("Units: hbar = kB = 1 (one common energy unit).")
TH = sb.slider("TH: hot-bath temperature", 0.5, 10.0, 4.0, 0.1, help="Temperature of the hot reservoir (energy units).")
TC = sb.slider("TC: cold-bath temperature", 0.1, 5.0, 1.0, 0.1, help="Temperature of the cold reservoir. Must be below TH.")
wH = sb.slider("wH: hot energy gap", 0.2, 6.0, 2.0, 0.1, help="Size of the qubit energy gap during the hot isochore.")
wC = sb.slider("wC: cold energy gap", 0.1, 5.0, 1.0, 0.1, help="Gap during the cold isochore. Must be below wH.")
J = sb.slider("J: qubit-qubit interaction", 0.0, 2.0, 0.5, 0.05, help="Coupling strength of the two qubits (kept fixed while the gap changes).")
kappa = sb.slider("kappa: thermal contact rate", 0.05, 20.0, 1.0, 0.05,
                  help="How strongly each bath pulls the qubits towards thermal equilibrium during a stroke of duration 1. "
                       "Large kappa = almost complete thermalisation (the starting state is forgotten).")
gamma = sb.slider("gamma: decoherence rate", 0.0, 5.0, 0.2, 0.05,
                  help="Strength of dephasing noise on each qubit (sigma_z noise). gamma = 0 means no extra noise.")
sb.subheader("Initial states")
p_ent = sb.slider("p: entanglement parameter", 0.0, 1.0, 0.5, 0.05,
                  help="Entangled state sqrt(1-p)|00> + sqrt(p)|11>. p = 0.5 is the Bell state; p = 0 or 1 is a product state.")
ref_kind = sb.selectbox("Reference state", sm.REFERENCES, help="The state the entangled state is compared with.")
g_max = sb.number_input("Largest gamma in the decoherence sweep", min_value=0.5, max_value=20.0, value=5.0, step=0.5)

params = {"TH": TH, "TC": TC, "wH": wH, "wC": wC, "J": J, "kappa": kappa, "gamma": gamma}
snapshot = (tuple(params.items()), p_ent, ref_kind, g_max)

# =============================================================================
# SECTION 3: Run simulation
# =============================================================================
problems = q.check_params(params)
for msg in problems:
    st.error(msg)
run = sb.button("Run simulation", type="primary", disabled=bool(problems))

if run and not problems:
    try:
        with st.spinner("Running the four-stage cycle and the sweeps..."):
            rho_ent = q.state_p(p_ent)
            rho_ref = sm.reference_state(ref_kind, params, p_ent)
            res = {
                "snapshot": snapshot, "params": dict(params), "p_ent": p_ent, "ref_kind": ref_kind,
                "ent": q.simulate(params, rho_ent), "ref": q.simulate(params, rho_ref),
                "memory": q.memory_after_hot_isochore(params, rho_ent, rho_ref),
                "df_p": sm.sweep_entanglement(params, 41),
                "df_g": sm.sweep_decoherence(params, np.linspace(0.0, g_max, 41), p_ent),
            }
            series = {"entangled": sm.work_per_cycle(params, rho_ent),
                      "reference": sm.work_per_cycle(params, rho_ref)}
            if ref_kind != sm.REF_TWIN:
                series["dephased copy"] = sm.work_per_cycle(params, sm.reference_state(sm.REF_TWIN, params, p_ent))
            res["series"] = series
            res["text"] = sm.explain_results(params, res["ent"], res["ref"], ref_kind, res["df_g"], res["memory"])
            st.session_state["result"] = res
    except q.ValidationError as err:
        st.error(f"The simulation stopped because a physical check failed: {err}")

res = st.session_state.get("result")
if res is None:
    st.info("Set the inputs in the sidebar and press **Run simulation**.")
    st.stop()
if res["snapshot"] != snapshot:
    st.warning("The inputs have changed since the last run. The results below belong to the OLD inputs. "
               "Press **Run simulation** to update them.")

P, ent, ref = res["params"], res["ent"], res["ref"]
st.header("Results of the last run")
st.caption(f"Run with TH={P['TH']}, TC={P['TC']}, wH={P['wH']}, wC={P['wC']}, J={P['J']}, kappa={P['kappa']}, "
           f"gamma={P['gamma']}, p={res['p_ent']}. Reference state: {res['ref_kind']}. Units: hbar = kB = 1.")

# ---- energy at each stage
stage_names = ["Start of cycle", "After 1 (hot isochore)", "After 2 (expansion)", "After 3 (cold isochore)", "After 4 (compression)"]
st.subheader("Energy at each stage:  U = Tr(rho H)")
st.dataframe(pd.DataFrame({
    "Stage": stage_names,
    "Entangled, cycle 1": ent["first"]["U"], "Reference, cycle 1": ref["first"]["U"],
    "Entangled, settled": ent["steady"]["U"], "Reference, settled": ref["steady"]["U"],
}).round(5), hide_index=True)

# ---- heat, work, efficiency
carnot = q.carnot_efficiency(P["TH"], P["TC"])
otto = q.otto_efficiency(P["wH"], P["wC"])


def fmt_eta(c):
    return f"{c['eta']:.4f}" if np.isfinite(c["eta"]) else "not defined"


rows = [("QH (heat from hot bath)", "QH"), ("QC (heat exchanged with cold bath)", "QC"),
        ("Wout (net work by the qubits)", "W"), ("Delta U over the cycle", "dU"),
        ("Energy-balance error QH+QC-W-DeltaU", "eps")]
table = {"Quantity": [r[0] for r in rows]}
for label, sim, key in (("Entangled, cycle 1", ent, "first"), ("Reference, cycle 1", ref, "first"),
                        ("Entangled, settled", ent, "steady"), ("Reference, settled", ref, "steady")):
    table[label] = [f"{sim[key][r[1]]:.5f}" if r[1] != "eps" else f"{sim[key][r[1]]:.1e}" for r in rows]
table["Quantity"] += ["Efficiency eta = Wout/QH", "Concurrence of the starting state", "Operating as heat engine?"]
for label, sim, key in (("Entangled, cycle 1", ent, "first"), ("Reference, cycle 1", ref, "first"),
                        ("Entangled, settled", ent, "steady"), ("Reference, settled", ref, "steady")):
    table[label] += [fmt_eta(sim[key]), f"{sim[key]['conc'][0]:.3f}", "Yes" if sim[key]["is_engine"] else "No"]
st.subheader("Heat, work and efficiency")
st.dataframe(pd.DataFrame(table), hide_index=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Carnot efficiency 1 - TC/TH", f"{carnot:.4f}")
m2.metric("Uncoupled-qubit Otto 1 - wC/wH", f"{otto:.4f}")
m3.metric("Cycles until the engine settles", f"{ent['n_cycles']}" + ("" if ent["converged"] else " (not converged)"))
m4.metric("Starting-state difference kept after hot isochore", f"{res['memory']:.3f}")
st.markdown("**Settled cycle status:** " + ent["steady"]["status"])
st.markdown("**Cycle 1 status (entangled start):** " + ent["first"]["status"])
if not ent["first"]["is_engine"] or not ref["first"]["is_engine"]:
    st.caption("Cycle 1 is a transient: it starts from a prepared state, not a closed cycle (Delta U is not zero), "
               "so its efficiency is only reported when QH > 0 and Wout > 0 and may even exceed the Carnot limit "
               "because the starting energy is being used up. The settled cycle is the true engine operating point.")
if ent["steady"]["is_engine"] and ent["steady"]["eta"] > carnot + 1e-9:
    st.warning("Settled efficiency is above the Carnot limit. The decoherence noise acts as an extra energy source, "
               "so the two-bath Carnot bound does not apply here.")
st.caption(f"Numerical checks passed: all density matrices Hermitian, trace 1, positive (worst eigenvalue "
           f"{ent['state_errors']['min_eigenvalue']:.1e}); largest energy-balance error "
           f"{ent['max_energy_error']:.1e} (limit {q.TOL_ENERGY:.0e}).")
if abs(ent["first"]["U"][0] - ref["first"]["U"][0]) > 1e-9:
    st.info("The two starting states have different energy, so part of any difference in work is NOT due to entanglement.")

# =============================================================================
# SECTION 4: Graphs
# =============================================================================


def show(fig):
    st.pyplot(fig)
    pl.plt.close(fig)


st.header("Graphs")
show(pl.fig_cycle_energy(ent, ref, res["ref_kind"]))
show(pl.fig_heat_work(ent, ref))
df_p, df_g = res["df_p"], res["df_g"]
tag = {sm.REF_PLUS: "plus", sm.REF_00: "p00", sm.REF_TWIN: "twin"}[res["ref_kind"]]
left, right = st.columns(2)
with left:
    show(pl.fig_work_vs_p(df_p))
    if np.isfinite(df_g["eta_steady_ent"]).any():
        show(pl.fig_eff_vs_gamma(df_g, tag))
    else:
        st.info("No decoherence value in the sweep gives a working heat engine, so there is no efficiency curve.")
with right:
    if np.isfinite(df_p["eta_steady"]).any() or sm.first_cycle_physical(df_p).any():
        show(pl.fig_eff_vs_p(df_p))
    else:
        st.info("No value of p gives a valid heat-engine efficiency, so there is no efficiency curve.")
    show(pl.fig_advantage_vs_gamma(df_g))
n_excl = int((~sm.first_cycle_physical(df_p)).sum())
st.caption(f"Efficiency-vs-p graph: {n_excl} of {len(df_p)} cycle-1 points are left out (shaded) because the cycle "
           "is not a heat engine or its efficiency is above Carnot. They are still in the CSV file.")
show(pl.fig_work_per_cycle(res["series"], ent["steady"]["W"]))

# =============================================================================
# SECTION 5: Optimization
# =============================================================================
st.header("Optimization: best operating point on a grid")
st.caption("A simple grid search: every combination of two chosen parameters is simulated and the best VALID one is "
           "reported. It is the best point inside the tested range and grid only, not a universal optimum.")
DEFAULT_RANGE = {"TH": (1.0, 10.0), "TC": (0.2, 3.0), "wH": (1.0, 5.0), "wC": (0.3, 1.9),
                 "J": (0.0, 2.0), "kappa": (0.1, 10.0), "gamma": (0.0, 5.0)}
names = list(sm.PARAM_LABELS)
o1, o2, o3 = st.columns(3)
objective_label = o1.selectbox("Objective", ["Maximum work output", "Maximum efficiency"])
kind_label = o2.selectbox("Which cycle", ["Settled cycle (recommended)", "Cycle 1 from the entangled state"])
w_min = o3.number_input("Minimum work for the efficiency objective", min_value=0.0, value=0.01, step=0.005, format="%.3f")
a1, a2 = st.columns(2)
x_name = a1.selectbox("Parameter X", names, index=names.index("wC"), format_func=lambda n: sm.PARAM_LABELS[n])
y_name = a2.selectbox("Parameter Y", names, index=names.index("TC"), format_func=lambda n: sm.PARAM_LABELS[n])
b1, b2, b3, b4, b5 = st.columns(5)
x_lo = b1.number_input("X from", value=DEFAULT_RANGE[x_name][0], key=f"lo_{x_name}_x")
x_hi = b2.number_input("X to", value=DEFAULT_RANGE[x_name][1], key=f"hi_{x_name}_x")
y_lo = b3.number_input("Y from", value=DEFAULT_RANGE[y_name][0], key=f"lo_{y_name}_y")
y_hi = b4.number_input("Y to", value=DEFAULT_RANGE[y_name][1], key=f"hi_{y_name}_y")
n_grid = b5.number_input("Points per axis", min_value=3, max_value=40, value=12, step=1)

if st.button("Find best operating point"):
    if x_name == y_name:
        st.error("Choose two different parameters.")
    elif x_hi <= x_lo or y_hi <= y_lo:
        st.error("Each 'to' value must be larger than its 'from' value.")
    else:
        objective = "work" if objective_label.startswith("Maximum work") else "efficiency"
        cycle_kind = "steady" if kind_label.startswith("Settled") else "first"
        bar = st.progress(0.0)
        try:
            df_opt, best = sm.grid_search(P, x_name, np.linspace(x_lo, x_hi, int(n_grid)),
                                          y_name, np.linspace(y_lo, y_hi, int(n_grid)), q.state_p(res["p_ent"]),
                                          objective, cycle_kind, w_min, progress=bar.progress)
            base = sm.evaluate(P, q.state_p(res["p_ent"]), "baseline")
            st.session_state["opt"] = {"df": df_opt, "best": best, "objective": objective, "x": x_name, "y": y_name,
                                       "kind": cycle_kind, "base": base, "w_min": w_min}
        except q.ValidationError as err:
            st.error(f"Optimization stopped: {err}")
        bar.empty()

opt = st.session_state.get("opt")
if opt is not None:
    st.subheader("Optimization result")
    key = "W" if opt["objective"] == "work" else "eta"
    best = opt["best"]
    n_valid, n_all = int(opt["df"]["valid"].sum()), len(opt["df"])
    st.caption(f"Objective: {'maximum Wout' if key == 'W' else 'maximum efficiency with Wout >= ' + str(opt['w_min'])}; "
               f"{n_valid} of {n_all} grid points were valid engine operating points.")
    if best is None:
        st.warning("No valid heat-engine operating point exists on this grid. Try other ranges.")
    else:
        suffix = "steady" if opt["kind"] == "steady" else "first"
        base_val = opt["base"][("W_" if key == "W" else "eta_") + suffix]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Best {opt['x']}", f"{best[opt['x']]:.4g}")
        c2.metric(f"Best {opt['y']}", f"{best[opt['y']]:.4g}")
        c3.metric("Objective value", f"{best[key]:.5f}")
        st.write(f"At this point: Wout = {best['W']:.5f}, QH = {best['QH']:.5f}, QC = {best['QC']:.5f}, "
                 f"efficiency = {best['eta']:.5f}.  Your current inputs give {'Wout' if key == 'W' else 'efficiency'} = "
                 + (f"{base_val:.5f}." if np.isfinite(base_val) else "not defined (not a valid heat engine)."))
    show(pl.fig_opt_map(opt["df"], opt["x"], opt["y"], opt["objective"], best))
    with st.expander("Why are some grid points invalid?"):
        st.dataframe(opt["df"][~opt["df"]["valid"]][[opt["x"], opt["y"], "reason"]].head(200), hide_index=True)

# =============================================================================
# SECTION 6: Explanation of results (written from the numbers)
# =============================================================================
st.header("Explanation of these results")
for line in res["text"]:
    st.markdown("- " + line)

# =============================================================================
# SECTION 7: Download results
# =============================================================================
st.header("Download results")
d1, d2, d3 = st.columns(3)
d1.download_button("Entanglement sweep (CSV)", df_p.to_csv(index=False), "entanglement_sweep.csv", "text/csv")
d2.download_button("Decoherence sweep (CSV)", df_g.to_csv(index=False), "decoherence_sweep.csv", "text/csv")
if opt is not None:
    d3.download_button("Optimization grid (CSV)", opt["df"].to_csv(index=False), "optimization_grid.csv", "text/csv")
