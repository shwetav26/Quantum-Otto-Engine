"""
plotting.py
===========
All graphs.  Every function takes real calculated data and returns a matplotlib Figure.
Undefined values (for example efficiency when the engine does not work) are NOT drawn
as zero or joined by lines: they are left out and the excluded region is shaded.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quantum_engine as q
import simulation_manager as sm

C_ENT, C_REF, C_TWIN, C_GREY = "#c2334d", "#2a6f97", "#5a8f3d", "#777777"
TEX = {"TH": r"$T_H$", "TC": r"$T_C$", "wH": r"$\omega_H$", "wC": r"$\omega_C$",
       "J": r"$J$", "kappa": r"$\kappa$", "gamma": r"$\gamma$"}
STAGES = ["start", "after 1\nhot isochore", "after 2\nexpansion", "after 3\ncold isochore", "after 4\ncompression"]
UNIT = "(units: $\\hbar = k_B = 1$)"
SHORT = {sm.REF_PLUS: "product |+>|+>", sm.REF_00: "product |00>", sm.REF_TWIN: "dephased copy"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _runs(mask, x):
    """Contiguous runs of True in mask -> list of (x_start, x_end) padded by half a step."""
    runs, start = [], None
    step = (x[1] - x[0]) / 2 if len(x) > 1 else 0
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            runs.append((x[start] - step, x[end] + step))
            start = None
    return runs


def fig_cycle_energy(ent, ref, ref_label):
    """Internal energy U = Tr(rho H) at the start and after each of the 4 stages."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, key, title in ((axes[0], "first", "Cycle 1 (starts from the prepared state)"),
                           (axes[1], "steady", "Settled cycle (initial state forgotten)")):
        ax.plot(range(5), ent[key]["U"], "o-", color=C_ENT, label="entangled state")
        ax.plot(range(5), ref[key]["U"], "s--", color=C_REF, label="reference: " + SHORT.get(ref_label, ref_label))
        ax.set_xticks(range(5))
        ax.set_xticklabels(STAGES, fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("Internal energy $U=\\mathrm{Tr}(\\rho H)$ " + UNIT)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def fig_heat_work(ent, ref):
    """Bar chart of QH, QC, Wout and the energy change for the entangled and reference states."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    names = [r"$Q_H$", r"$Q_C$", r"$W_{out}$", r"$\Delta U_{cycle}$"]
    for ax, key, title in ((axes[0], "first", "Cycle 1"), (axes[1], "steady", "Settled cycle")):
        x = np.arange(4)
        for off, sim, col, lab in ((-0.2, ent, C_ENT, "entangled"), (0.2, ref, C_REF, "reference")):
            c = sim[key]
            ax.bar(x + off, [c["QH"], c["QC"], c["W"], c["dU"]], 0.38, color=col, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title + r": $Q_H+Q_C-W_{out}=\Delta U$")
    axes[0].set_ylabel("Energy per cycle " + UNIT)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_work_vs_p(df):
    """Wout versus the entanglement parameter p, plus the starting energy and the concurrence."""
    fig = plt.figure(figsize=(6.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df["p"], df["W_first"], "o-", ms=3, color=C_ENT, label=r"cycle 1 work $W_{out}$")
    ax.axhline(df["W_steady"].iloc[0], color=C_GREY, ls="--", label="settled-cycle work (same for every p)")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0.5, color=C_TWIN, ls=":", label="Bell state (p = 0.5)")
    ax.set_xlabel("p   in   $|\\psi(p)\\rangle=\\sqrt{1-p}\\,|00\\rangle+\\sqrt{p}\\,|11\\rangle$")
    ax.set_ylabel("$W_{out}$ " + UNIT)
    ax.set_title("Work output versus entanglement parameter")
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["p"], df["U_initial"], color=C_REF)
    ax2.set_xlabel("p"); ax2.set_ylabel("start energy $U_0$"); ax2.set_title("Energy at the start", fontsize=9)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["p"], df["conc_initial"], color=C_TWIN)
    ax3.set_xlabel("p"); ax3.set_ylabel("concurrence"); ax3.set_title("Entanglement (0 to 1)", fontsize=9)
    fig.tight_layout()
    return fig


def _eff_axes(ax, x, carnot, otto):
    ax.axhline(carnot, color="k", ls="--", lw=1, label=f"Carnot limit {carnot:.2f}")
    ax.axhline(otto, color=C_GREY, ls=":", lw=1.2, label=f"Otto $1-\\omega_C/\\omega_H$ (uncoupled qubits) = {otto:.2f}")
    ax.set_ylim(0, max(carnot, otto) * 1.12)


def fig_eff_vs_p(df):
    """Efficiency versus p, only where the engine really works.  Shaded = no valid first-cycle efficiency."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    p = df["p"].values
    ok = sm.first_cycle_physical(df).values
    _eff_axes(ax, p, df["carnot"].iloc[0], df["otto"].iloc[0])
    ax.plot(p[ok], df["eta_first"].values[ok], "o", ms=4, color=C_ENT, label="cycle 1 (transient, not closed): valid points only")
    steady = df["eta_steady"].values
    if np.isfinite(steady).any():
        ax.plot(p, steady, "-", color=C_REF, label="settled-cycle efficiency")
    for i, (a, b) in enumerate(_runs(~ok, p)):
        ax.axvspan(a, b, color="#bbbbbb", alpha=0.35, label="cycle 1: not an engine or above Carnot" if i == 0 else None)
    ax.set_xlabel("p"); ax.set_ylabel(r"efficiency $\eta=W_{out}/Q_H$")
    ax.set_title("Efficiency versus entanglement parameter")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_eff_vs_gamma(df_g, ref_tag):
    """Efficiency versus decoherence rate.  Settled-cycle curve plus valid cycle-1 points."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    g = df_g["gamma"].values
    _eff_axes(ax, g, df_g["carnot_ent"].iloc[0], df_g["otto_ent"].iloc[0])
    s = df_g["eta_steady_ent"].values
    if np.isfinite(s).any():
        ax.plot(g, s, "o-", ms=3, color=C_REF, label="settled cycle (same for any start state)")
    for col, color, lab in (("eta_first_ent", C_ENT, "cycle 1, entangled"), (f"eta_first_{ref_tag}", C_TWIN, "cycle 1, reference")):
        v = df_g[col].values
        ok = np.isfinite(v) & (v <= df_g["carnot_ent"].iloc[0] + 1e-9)
        if ok.any():
            ax.plot(g[ok], v[ok], "x", color=color, label=lab + " (transient; valid points only)")
    ax.set_xlabel(r"decoherence rate $\gamma$"); ax.set_ylabel(r"efficiency $\eta=W_{out}/Q_H$")
    ax.set_title("Efficiency versus decoherence")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def fig_advantage_vs_gamma(df_g):
    """dW = W(entangled) - W(reference) in cycle 1, for every reference choice."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    g = df_g["gamma"].values
    styles = ((sm.REF_PLUS, "plus", C_REF, r"vs product $|{+}{+}\rangle$ (same energy)"),
              (sm.REF_00, "p00", C_GREY, r"vs product $|00\rangle$"),
              (sm.REF_TWIN, "twin", C_TWIN, "vs dephased copy (coherence control)"))
    for kind, tag, col, lab in styles:
        d = df_g[f"dW_first_{tag}"].values
        ax.plot(g, d, "-", color=col, label=lab)
        cross, _ = sm.find_zero_crossings(g, d)
        for c in cross:
            ax.plot([c], [0], "o", color=col, mfc="white")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"decoherence rate $\gamma$")
    ax.set_ylabel(r"$\Delta W=W_{ent}-W_{ref}$ (cycle 1) " + UNIT)
    ax.set_title("Work difference (entangled minus reference) versus decoherence")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def fig_work_per_cycle(series, steady_value):
    """Work of cycle 1, 2, 3, ... for different starting states: shows the memory being erased."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = [C_ENT, C_REF, C_TWIN, C_GREY]
    for (label, works), col in zip(series.items(), colors):
        ax.plot(range(1, len(works) + 1), works, "o-", ms=3, color=col, label=label)
    ax.axhline(steady_value, color="k", ls="--", lw=1, label="settled value")
    ax.set_xlabel("cycle number"); ax.set_ylabel("$W_{out}$ per cycle " + UNIT)
    ax.set_title("Is the starting state remembered? Work in each cycle")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_opt_map(df, x_name, y_name, objective, best):
    """Map of the objective over the tested grid.  Grey cells = invalid operating points."""
    key = "W" if objective == "work" else "eta"
    xs, ys = np.sort(df[x_name].unique()), np.sort(df[y_name].unique())
    grid = np.full((len(ys), len(xs)), np.nan)
    for _, r in df[df["valid"]].iterrows():
        grid[np.searchsorted(ys, r[y_name]), np.searchsorted(xs, r[x_name])] = r[key]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    dx = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
    dy = (ys[1] - ys[0]) / 2 if len(ys) > 1 else 0.5
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap,
                   extent=[xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy])
    fig.colorbar(im, ax=ax, label=r"$W_{out}$" if key == "W" else r"efficiency $\eta$")
    if best is not None:
        ax.plot([best[x_name]], [best[y_name]], "*", ms=16, color="white", mec="k", label="best valid point")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel(TEX.get(x_name, x_name)); ax.set_ylabel(TEX.get(y_name, y_name))
    ax.set_title("Grid search (grey = invalid operating point)")
    ax.grid(False)
    fig.tight_layout()
    return fig
