# Two-Qubit Quantum Otto Engine Simulator

**Work Extraction from Entangled Systems: Efficiency and Performance of a Two-Qubit Quantum Otto Engine Under Decoherence**
(Engineering Physics project, Review 2 / final)

A small Python program that simulates the four strokes of a quantum Otto cycle for two qubits, compares an
entangled starting state with non-entangled ones, adds decoherence, and searches a grid for the best operating point.

---------------------------------------------------------------------------------------------------------

## 1. Setup and run (Windows PowerShell)

Install Python 3.10 or newer from python.org (tick "Add Python to PATH"), then in PowerShell:

```powershell
cd path\to\quantum_otto_project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at http://localhost:8501. Press **Run simulation** in the left sidebar.

Run the automated tests (no extra install needed):

```powershell
python -m unittest discover -s tests -v
```

Regenerate every graph / CSV used in the report:

```powershell
python make_report_assets.py
```

**Troubleshooting**

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again. |
| `streamlit is not recognized` | Activate the venv first, or use `python -m streamlit run app.py`. |
| Page shows "inputs have changed" | You moved a slider after the last run. Press **Run simulation** again. |
| Red error "wH must be greater than wC" | The inputs are not physical (see section 5). Fix the sliders. |
| Optimization says "No valid heat-engine operating point" | Every grid point failed a check. Widen/shift the ranges. |

**Honest note on testing:** the 75 automated tests were run on Python 3.12.3 / NumPy 2.4.4 / SciPy 1.17.1 /
Matplotlib 3.10.8 / pandas 3.0.2. The Streamlit page itself could **not** be opened in the environment where the code
was written (no network, Streamlit not installable there). `app.py` was run against a stand-in `streamlit`
module (tests/fake_streamlit.py), which proves the app logic runs without Python errors, but **not** that the real
pages look right. Please do the manual check in section 8 once on your laptop.

---------------------------------------------------------------------------------------------------------

## 2. Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit interface (inputs, Run button, tables, graphs, optimization, explanation, CSV download). |
| `quantum_engine.py` | All physics: Hamiltonian, states, thermal state, concurrence, decoherence, the four strokes, heat/work/efficiency, checks. |
| `simulation_manager.py` | Entanglement sweep, decoherence sweep, zero-crossing finder, grid-search optimizer, automatic explanation. |
| `plotting.py` | All graphs (Matplotlib). |
| `make_report_assets.py` | Re-creates the graphs, CSVs and `results/key_numbers.json` for the presentation. |
| `tests/` | 75 automated tests (`unittest`). `fake_streamlit.py` is only a test helper. |
| `results/` | Graphs, CSVs, key numbers and the saved test output from the final run. |
| `TESTING_REPORT.md`, `VIVA_PREP.md` | Validation report and viva questions with answers. |

---------------------------------------------------------------------------------------------------------

## 3. The model (internal specification)

**Units.** hbar = kB = 1. Energies, energy gaps, temperatures and rates all use one "energy unit"; time is in
1/energy. Each isochore lasts TAU = 1 time unit. There are no SI conversions anywhere.

**Basis.** |00>, |01>, |10>, |11> (4-dimensional). State = 4x4 density matrix rho.

**Hamiltonian** (the simplest one in which the energy levels do not all scale with the gap):

H(w) = -(w/2) (sz (x) I + I (x) sz) - J (sx (x) sx)

* w is the energy-gap parameter (wH on the hot stroke, wC on the cold stroke, wH > wC > 0).
* J >= 0 is a *fixed* interaction between the qubits. It does **not** change with w.
* Energy levels (exact): -sqrt(w^2+J^2), -J, +J, +sqrt(w^2+J^2). Levels never cross for w > 0.
* Eigenvectors are written by hand in `level_basis` and checked against NumPy in the tests.
* |0> is the lower level of each qubit. The Bell state |Phi+> = (|00>+|11>)/sqrt2 has energy exactly -J.

*Change from Review 1.* The candidate H had the interaction proportional to w (lambda*w). We tested it first:
then H(wC) = (wC/wH) H(wH) exactly, and **every** cycle has efficiency exactly 1 - wC/wH whatever the state,
coupling or noise. Efficiency-vs-entanglement and efficiency-vs-decoherence would be flat lines and the research
question about efficiency could not be answered. A *fixed* J (the standard choice in the two-qubit Otto-engine
literature, e.g. Zhang et al. 2007) removes this. The sign of the terms was chosen so that |0> is the lower level and
|Phi+> is a low-energy state; this is the same physics as the Review 1 form after relabelling basis states.
(For J = 0 the model reduces to two independent qubits, giving the exact textbook result 1 - wC/wH; this is a test.)

**Initial states.**
* Entangled family: |psi(p)> = sqrt(1-p)|00> + sqrt(p)|11>. p = 0.5 is the Bell state; p = 0 and p = 1 are product states.
* Reference states (choose one in the app):
  1. |+>|+> (product state; **same energy as the Bell state**, -J) - default, the fairest comparison.
  2. |00> (product state; much lower energy, so it differs in energy as well as in entanglement).
  3. "Dephased copy" of the entangled state: same populations of the energy levels, all coherence removed. This is a
     *control for coherence*; it is **not guaranteed to be a product state** (its concurrence is reported).
* Entanglement measure: Wootters concurrence C (0 = not entangled, 1 = maximally entangled). Superposition alone does not give C > 0.

**Thermal state** (Gibbs): rho_th = exp(-H/T) / Tr exp(-H/T), computed from the eigenvalues (shifted by the lowest
one so that nothing overflows). It depends on both H and T. A Bell state is *not* a thermal state.

**Environment (one equation, used on both isochores):**

d rho/dt = -i[H, rho]  +  kappa (rho_th - rho)  +  gamma * sum_{j=1,2} ( Z_j rho Z_j - rho )

* `-i[H,rho]`: ordinary quantum evolution.
* `kappa (rho_th - rho)`: **thermalisation / partial thermalisation.** The bath pulls the state towards the Gibbs
  state at rate kappa. After a stroke of duration 1 the fraction e^(-kappa) of the old state is still there (for gamma = 0 this is exact).
  This is a valid completely-positive evolution (a mixture of "do nothing" and "replace by rho_th").
* `gamma(...)`: **decoherence** = random phase kicks (dephasing) on each qubit, collapse operators sqrt(gamma)*sigma_z on qubit 1 and 2.
  Coherences between basis states that differ on k qubits decay as exp(-2 k gamma t).
* **Does the noise commute with H?** Only for J = 0. For J > 0 sigma_z does not commute with the sx*sx term, so the noise can
  change the energy of the qubits. We keep this on purpose (it is the realistic qubit noise) but report it: with noise the
  system also exchanges energy with the noise environment, so the two-bath Carnot bound is not guaranteed (see section 6).
* The adiabatic strokes are ideal (no noise, no friction).

**Which partial-thermalisation choice, and why.** With *full* thermalisation (kappa large) the hot bath resets the state to
rho_th and **erases every difference between initial states before any work is extracted**; the comparison would be
meaningless. The app therefore uses finite contact (default kappa = 1 -> 63 % thermalised per stroke), prints the fraction of
the initial-state difference that survives the hot isochore (default 0.33), and the tests show that for large kappa this
fraction goes to 0.

**The four strokes (cycle starts with H(wH) and a state rho0).**

| Stage | What changes | Calculation |
|---|---|---|
| 1 Hot isochore | H = H(wH) fixed, contact with hot bath (TH) + noise | rho1 = exp(L_H * 1) rho0; QH = Tr[H_H (rho1 - rho0)] |
| 2 Adiabatic expansion | wH -> wC, no heat | Each energy level k at wH becomes level k at wC (state is carried by A = V_C V_H^T); populations unchanged; W2 = sum_k p_k (E_k^H - E_k^C) |
| 3 Cold isochore | H = H(wC) fixed, cold bath (TC) + noise | rho3 = exp(L_C * 1) rho2; QC = Tr[H_C (rho3 - rho2)] |
| 4 Adiabatic compression | wC -> wH, no heat | W4 = sum_k p_k (E_k^C - E_k^H) |

The output state of each stage is the input of the next, and the final state of cycle n is the start of cycle n+1.
Dynamical phases during the adiabatic strokes are ignored (they cannot change any energy).

**Sign convention.** Q > 0: heat flows into the qubits. W > 0: work done by the qubits (Wout).
First law, every cycle: QH + QC - Wout = U_end - U_start, with U = Tr(rho H).
The program computes Q from the isochores (traces) and W from level populations (a different formula), so the
energy-balance error eps is a real check. It must be < 1e-9, otherwise the program stops with an error.

**Efficiency.** eta = Wout/QH only if QH > 0 and Wout > 0; otherwise it is "not defined" and the cycle is labelled
"Not a heat engine" with a reason. Carnot: eta_C = 1 - TC/TH. "Uncoupled-qubit Otto" = 1 - wC/wH (exact only for J = 0).

**Two kinds of cycle (important).** An engine run for many cycles forgets its starting state: the cycle "settles" (limit cycle,
Delta U = 0, W = QH + QC, found automatically; usually 10-20 cycles). We therefore report
* **cycle 1** - starts from the prepared (entangled / reference) state. This is where initial-state differences can appear. It is a *transient*, not a closed
  cycle (Delta U != 0), so its "efficiency" can exceed Carnot (it burns the starting energy). It is shown only when QH > 0 and Wout > 0, and efficiency plots exclude (shade and count) cycle-1 points that are undefined or above Carnot - nothing is clipped; all values are in the CSV.
* **settled cycle** - the real engine operating point. Independent of the starting state (tested).

**Entangled-vs-reference advantage.** dW = Wout(entangled) - Wout(reference) in cycle 1 (and d eta only where both are defined).
It is called an *entanglement* advantage only when the reference has the same energy; otherwise part of dW is stored energy.

---------------------------------------------------------------------------------------------------------

## 4. Inputs and outputs of the app

Inputs (sidebar): TH, TC, wH, wC, J, kappa, gamma, p, reference state, largest gamma in the sweep.
Optimization inputs: objective, which cycle, two parameters with ranges and grid size, minimum work.

Outputs: energy at each stage; QH, QC, Wout, Delta U, eta, concurrence, energy-balance error, engine status for cycle 1 and settled cycle;
Carnot and uncoupled-Otto efficiencies; number of cycles to settle; fraction of the initial-state difference surviving the hot isochore;
7 graphs; optimization result and map; an automatic explanation built from the numbers; three CSV downloads.

## 5. Valid input ranges (checked before every run)
TH > TC > 0; wH > wC > 0; J >= 0; kappa > 0 (no heat contact = no engine); gamma >= 0; all numbers finite. Invalid combinations are refused with a message (and
marked invalid in the optimization grid) rather than simulated.

## 6. Optimization (grid search)
Pick two parameters, a range for each and a grid size. Every grid point is simulated. A point is **valid** only if the
parameters are physical, QH > 0 and Wout > 0, the settled cycle converged and eta <= Carnot. Two separate objectives:
*maximum Wout*, and *maximum eta among points with Wout >= a minimum work* (default 0.01). They give different points; there is no weighted score.
The result is the best valid point **inside the tested grid**, not a universal optimum.

## 7. Known limitations
* Strokes 2 and 4 are ideal; no quantum friction, no noise during them. Strokes 1 and 3 have a fixed duration of 1.
* The thermalisation is the simple relaxation-to-Gibbs-state model, not a microscopic bath (Davies/Lindblad with jump operators between levels).
* sigma_z dephasing with J > 0 is an extra, non-thermal energy source. With it the Carnot/Clausius bounds can fail (see TESTING_REPORT.md).
* Only two qubits and one Hamiltonian family. Results are statements about this model, not about all quantum engines.
* This is a computational study; nothing here is an experimental demonstration of a quantum advantage.

## 8. Manual check to do once on your laptop (about 2 minutes)
1. `streamlit run app.py`, press **Run simulation** with defaults. You should see: Carnot 0.75; settled efficiency 0.4045; settled Wout 0.0874; ~14 cycles; memory 0.328.
2. Set wC = 3 (above wH): a red error appears and the Run button is disabled.
3. Change TH after a run: a yellow "inputs have changed" banner appears.
4. Run the optimization with the defaults: best work about 0.29 at wC about 0.88, TC = 0.2 (grid 12x12).
5. Click the three download buttons and open the CSVs.
