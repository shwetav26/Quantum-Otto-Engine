# Testing and Validation Report

## 1. What was actually run
* Command: `python -m unittest discover -s tests -v`  (saved in `results/test_output.txt`)
* Environment: Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8, pandas 3.0.2 (Linux sandbox).
* **Result of the final run: 75 tests run, 75 passed, 0 failed, 0 errors** (47 physics, 22 sweeps/optimizer/graphs, 6 app smoke tests).
* `python make_report_assets.py` was also run; all graphs, CSVs and `results/key_numbers.json` come from it.

## 2. What could NOT be run here (and how to check it)
| Not run | Why | Manual procedure |
|---|---|---|
| The real Streamlit web page | Streamlit could not be installed in the sandbox (no network). | README section 8 (five quick checks). |
| Windows PowerShell commands | Sandbox is Linux. | Commands in README section 1 use only standard Python tools. |
| Screenshots of the interface | Needs a browser + Streamlit. | Take screenshots yourself (Win+Shift+S) and place them on the presentation's interface slide. |

The six `test_app_smoke.py` tests run `app.py` against a small stand-in for `streamlit` (`tests/fake_streamlit.py`). They show that every
section of the app executes without Python errors, that inputs reach the calculation, that the CSV has 41 rows and the right columns,
that invalid inputs are refused and that stale results are flagged. They do **not** test Streamlit's own rendering.

## 3. Numerical tolerances
| Quantity | Tolerance | Reason |
|---|---|---|
| Hermitian error of rho | 1e-10 | matrix exponential round-off is ~1e-16 |
| Trace error | 1e-10 | same |
| Smallest eigenvalue | >= -1e-9 | round-off in the exponential |
| Energy balance eps per cycle | 1e-9 | actual worst value in tests is ~1e-16 |
| Settled cycle (state change per cycle) | 1e-12 | at most 3000 cycles allowed |
| Eigenvalues below 1e-14 in the concurrence | set to 0 | they are round-off; their square roots would add ~1e-9 error |
| Carnot comparison | +1e-9 | floating-point margin |
No state is ever silently repaired; a violation stops the run with an error message.

## 4. Test groups and the independent references used
**Mathematical tests (physics file).** Normalisation and purity of all pure states; Hermitian / trace-1 / positive checks on every state created
and on every stage state of 20 random-parameter cycles; spectrum of H against NumPy and the closed form (-sqrt(w^2+J^2), -J, J, +sqrt(w^2+J^2));
hand-written eigenvectors diagonalise H and are orthonormal; levels never cross; Gibbs state has trace 1, equals exp(-H/T)/Tr (SciPy `expm`), does not overflow at T = 0.01,
is diagonal in the energy basis with lower energy more populated; internal energy is real; efficiency definition (NaN when QH <= 0 or W <= 0);
parameter / NaN / infinity validation.

**Entanglement measure.** Bell state C = 1; |00>, |+>|+>, I/4 give C = 0; superposition `|0>(|0>+|1>)` gives C = 0 (superposition is not entanglement);
`sqrt(1-p)|00>+sqrt(p)|11>` gives 2 sqrt(p(1-p)); Werner states give max(0,(3v-1)/2). All are closed-form results.

**Heat/work/First law.** For 60 random parameter sets and random p, QH + QC - W - Delta U < 1e-9 (cycle 1 and settled cycle). Signs: for the default engine QH > 0, QC < 0, W > 0, Clausius sum > 0;
a hot starting state (|11>) has QH < 0 and a cold one (|00>) has QH > 0. The adiabatic strokes preserve level populations and exchange no heat (W2 = U1 - U2).

**Independent verification (not "same function twice").**
1. The 16x16 evolution is compared with direct integration of the master equation using SciPy `solve_ivp` and plain matrix products (agreement < 1e-8).
2. For gamma = 0 the isochore is compared with the hand-derived formula e^(-kappa) U rho U^+ + (1-e^(-kappa)) rho_th (agreement 1e-12).
3. Pure-dephasing decay laws: |00><11| decays as e^(-4 gamma t), |00><01| as e^(-2 gamma t), populations unchanged, concurrence of dephased Bell = e^(-4 gamma t).
4. The settled state is recomputed by a separate linear-algebra solver (fixed point of the whole-cycle superoperator built from NumPy eigenvectors for the level ordering) and agrees to 1e-8; the settled work computed from it matches.
5. J = 0 (two independent qubits): settled efficiency equals 1 - wC/wH to 9 decimals, with and without noise.

**Limiting cases.** Zero decoherence (continuous as gamma -> 1e-9); increasing decoherence (settled work falls monotonically for the default inputs); strong thermal contact kappa = 20 erases the
initial state (memory < 1e-8, first-cycle work identical); settled cycle is identical for 4 different starting states; TH = TC and other invalid combinations are refused; a cycle with no net work
(TH = 1.2, TC = 1) is labelled "Not a heat engine" with no efficiency.

**Sweeps / optimizer / graphs.** 41-point entanglement sweep: ordered x, Bell point at p = 0.5 equals a direct simulation, concurrence 0/1/0; decoherence sweep: 41 rows, gamma = 0 twin difference = 0,
energy-matched reference really has equal energy, CSV round-trip identical; optimizer best point equals the true maximum of valid points and is recomputed independently; invalid grid cells are rejected without being simulated;
work-optimum and efficiency-optimum are different points; minimum-work constraint respected; empty-valid grid returns None; progress callback reached 100 %; plotted lines equal the computed data; all axes have titles and labels;
efficiency graph contains no NaN points and exactly the valid cycle-1 points.

## 5. Problems found by the tests and fixed (root cause, not hidden)
1. **Concurrence error of 3.5e-9** for pure states: square roots of round-off eigenvalues (~1e-17). Rewritten with singular values and a 1e-14 eigenvalue cut-off.
2. **A wrong physics assumption in my own test**: I expected eta <= 1 - wC/wH for all J. For J > 0 this is false (levels E-J scale less than proportionally with w); only Carnot is general.
   Test corrected, an explicit test of the J > 0 behaviour added, and the app/graph label changed to "uncoupled-qubit Otto".
3. **Fake zero-crossing at gamma = 0**: the dephased-copy difference is exactly 0 at gamma = 0 and was reported as a threshold. The finder now ignores zero points; only real sign changes count.
4. A stray no-op line in `app.py` and a miscounted expectation in an app test were removed/corrected.

## 6. Scientific findings from the validation (important for the viva)
* **Energy balance** error is ~1e-16 in every test; **all states stay valid**.
* **Coherence alone does nothing without noise**: with gamma = 0 the Bell state and its dephased copy give *identical* work in every cycle (difference < 1e-12). Only the populations of the energy levels matter.
* **The settled engine forgets the start state** (identical to 1e-9). Any initial-state advantage lives only in the first few cycles.
* **Entanglement does not survive stage 1** at the default inputs: concurrence of the Bell state is 1 at the start and 0 after the hot isochore. So during the work strokes the state is separable.
* **Bounds.** Without noise (gamma = 0): in a seeded random test of 2000 parameter sets there were 1662 working engines, **0** Carnot violations and **0** Clausius (second-law) violations.
  With noise: 1436 working engines, 0 Carnot violations in that seed, but **128** negative Clausius sums (-QH/TH - QC/TC < 0). An earlier exploratory run with another seed
  found a Carnot violation (eta 0.710 against 0.616) at TH 1.376, TC 0.528, wH 4.189, wC 2.477, J 1.304, kappa 3.601, gamma 0.427; the same point with gamma = 0 obeys Carnot (this case is pinned in a test).
  Interpretation: sigma_z noise does not commute with the J term, so it acts as an additional, non-thermal energy source; the two-bath Carnot bound then need not hold. The program flags such points and the optimizer excludes them. It does not clip them.
* **Limitations of the validation:** the tests prove the code solves *this model* correctly and consistently; they cannot prove that the model describes any real device.
