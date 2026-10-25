"""
quantum_engine.py
=================
Physics of a two-qubit quantum Otto engine (all the important maths is here).

UNITS: we use hbar = kB = 1.  Energies, frequencies, temperatures and 1/time
all share one "energy unit".  Time is measured in 1/(energy unit).

MODEL (see README for the full explanation)
  Hamiltonian   H(w) = -(w/2)(Z1 + Z2) - J (X1 X2)
  Bath contact  d(rho)/dt = -i[H,rho] + kappa (rho_thermal - rho)      <- thermalisation
                            + gamma * sum_j (Z_j rho Z_j - rho)        <- decoherence
  Strokes       1 hot isochore, 2 adiabatic expansion,
                3 cold isochore, 4 adiabatic compression
Basis order of the 4x4 matrices: |00>, |01>, |10>, |11>.
"""
import numpy as np
from scipy.linalg import expm, eigh

# ----------------------------------------------------------------------------
# Constants and tolerances (every tolerance is documented in the README)
# ----------------------------------------------------------------------------
TAU = 1.0            # duration of each isochore stroke (time unit)
TOL_HERM = 1e-10     # max allowed |rho - rho^dagger|
TOL_TRACE = 1e-10    # max allowed |Tr(rho) - 1|
TOL_POS = 1e-9       # smallest allowed eigenvalue is -TOL_POS
TOL_ENERGY = 1e-9    # max allowed energy-balance error per cycle
TOL_STEADY = 1e-12   # cycle is "closed" when rho changes by less than this
MAX_CYCLES = 3000    # safety limit on the number of cycles we iterate

# Pauli matrices and identity (complex type so density matrices stay complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

Z1 = np.kron(SZ, I2)       # sigma_z on qubit 1
Z2 = np.kron(I2, SZ)       # sigma_z on qubit 2
XX = np.kron(SX, SX)       # sigma_x (x) sigma_x  (the interaction)

DEFAULT_PARAMS = {
    "TH": 4.0,      # hot-bath temperature
    "TC": 1.0,      # cold-bath temperature
    "wH": 2.0,      # energy-gap parameter during the hot isochore
    "wC": 1.0,      # energy-gap parameter during the cold isochore
    "J": 0.5,       # interaction strength between the qubits (fixed, does not change with w)
    "kappa": 1.0,   # thermal-contact rate with each bath (strength of thermalisation)
    "gamma": 0.2,   # decoherence rate (dephasing of each qubit)
}


class ValidationError(ValueError):
    """Raised when a parameter or a quantum state is not physically valid."""


# ----------------------------------------------------------------------------
# 1. Parameter and state validation
# ----------------------------------------------------------------------------
def check_params(p):
    """Return a list of problems with the parameter dictionary (empty list = OK)."""
    problems = []
    for key in DEFAULT_PARAMS:
        if key not in p:
            problems.append(f"Missing parameter {key}.")
        elif not np.isfinite(p[key]):
            problems.append(f"{key} must be a finite number.")
    if problems:
        return problems
    if p["TC"] <= 0:
        problems.append("TC must be greater than 0.")
    if p["TH"] <= p["TC"]:
        problems.append("TH must be greater than TC.")
    if p["wC"] <= 0:
        problems.append("wC must be greater than 0.")
    if p["wH"] <= p["wC"]:
        problems.append("wH must be greater than wC.")
    if p["J"] < 0:
        problems.append("J must be 0 or positive.")
    if p["kappa"] <= 0:
        problems.append("kappa must be greater than 0 (the engine needs heat contact).")
    if p["gamma"] < 0:
        problems.append("gamma must be 0 or positive.")
    return problems


def check_state(rho, name="rho"):
    """Check Hermitian, trace one, positive semidefinite, finite.
    Raises ValidationError if something is wrong (we never silently repair a state).
    Returns the three error sizes so they can be reported."""
    if not np.all(np.isfinite(rho)):
        raise ValidationError(f"{name} contains NaN or infinity.")
    herm = np.max(np.abs(rho - rho.conj().T))
    trace = abs(np.trace(rho).real - 1.0)
    min_eig = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2))
    if herm > TOL_HERM:
        raise ValidationError(f"{name} is not Hermitian (error {herm:.2e}).")
    if trace > TOL_TRACE:
        raise ValidationError(f"{name} does not have trace 1 (error {trace:.2e}).")
    if min_eig < -TOL_POS:
        raise ValidationError(f"{name} is not positive semidefinite (min eigenvalue {min_eig:.2e}).")
    return {"hermitian_error": herm, "trace_error": trace, "min_eigenvalue": min_eig}


# ----------------------------------------------------------------------------
# 2. Hamiltonian and its energy levels
# ----------------------------------------------------------------------------
def hamiltonian(omega, J):
    """H(w) = -(w/2)(Z1+Z2) - J X1X2  (4x4).  |0> is the lower level of each qubit."""
    return -0.5 * omega * (Z1 + Z2) - J * XX


def level_basis(omega, J):
    """Exact eigenvectors and energies of H(omega) written by hand.
    Returns (V, E): column k of V is the eigenvector with energy E[k],
    ordered  E = [-sqrt(w^2+J^2), -J, +J, +sqrt(w^2+J^2)].
    Two blocks decouple: {|00>,|11>} and {|01>,|10>}.
    Because the columns keep the SAME ORDER for every omega, 'level k at hot'
    is mapped to 'level k at cold' by the adiabatic stroke."""
    big = np.hypot(omega, J)               # sqrt(w^2 + J^2)
    theta = np.arctan2(J, omega)           # tan(theta) = J / w
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    r = 1 / np.sqrt(2)
    V = np.zeros((4, 4))
    V[0, 0], V[3, 0] = c, s                # lowest level: c|00> + s|11>
    V[1, 1], V[2, 1] = r, r                # -J level: (|01>+|10>)/sqrt2
    V[1, 2], V[2, 2] = r, -r               # +J level: (|01>-|10>)/sqrt2
    V[0, 3], V[3, 3] = -s, c               # highest level: -s|00> + c|11>
    E = np.array([-big, -J, J, big])
    return V, E


def gibbs_state(H, T):
    """Thermal (Gibbs) state exp(-H/T)/Z.  Energies are shifted by the lowest one
    before exponentiating so the numbers never overflow."""
    energies, vectors = eigh(H)
    weights = np.exp(-(energies - energies.min()) / T)
    weights /= weights.sum()
    return (vectors * weights) @ vectors.conj().T


def internal_energy(rho, H):
    """U = Tr(rho H).  It must be real; we check this and return the real number."""
    u = np.trace(rho @ H)
    if abs(u.imag) > 1e-9:
        raise ValidationError(f"Internal energy has an imaginary part {u.imag:.2e}.")
    return float(u.real)


# ----------------------------------------------------------------------------
# 3. Quantum states and entanglement
# ----------------------------------------------------------------------------
def ket_to_rho(ket):
    ket = np.asarray(ket, dtype=complex)
    ket = ket / np.linalg.norm(ket)                # make sure <psi|psi> = 1
    return np.outer(ket, ket.conj())


def state_p(p):
    """|psi(p)> = sqrt(1-p)|00> + sqrt(p)|11>.  p=0.5 is the Bell state."""
    if not (0.0 <= p <= 1.0):
        raise ValidationError("p must be between 0 and 1.")
    return ket_to_rho([np.sqrt(1 - p), 0, 0, np.sqrt(p)])


def state_bell():
    return state_p(0.5)                            # (|00>+|11>)/sqrt2


def state_product_00():
    return ket_to_rho([1, 0, 0, 0])                # both qubits in |0>


def state_plus_plus():
    return ket_to_rho([1, 1, 1, 1])                # |+>|+>, a product state


def dephased_copy(rho, omega, J):
    """Keep only the energy-level populations of rho (w.r.t. H(omega)) and throw
    away all coherence between levels.  Same populations => same energy content."""
    V, _ = level_basis(omega, J)
    pops = np.real(np.diag(V.T @ rho @ V))
    return (V * pops) @ V.T


def concurrence(rho):
    """Wootters concurrence of a two-qubit state: 0 = not entangled, 1 = maximally entangled.
    C = max(0, s1 - s2 - s3 - s4), where s_i are the singular values of
    sqrt(rho) (Y x Y) conj(sqrt(rho)) (Y x Y).  Eigenvalues of rho below 1e-14 are rounding noise
    and are set to zero (otherwise their square roots would add errors of about 1e-9)."""
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2)
    w = np.where(w > 1e-14, w, 0.0)
    root = (v * np.sqrt(w)) @ v.conj().T
    yy = np.kron(SY, SY)
    s = np.linalg.svd(root @ yy @ root.conj() @ yy, compute_uv=False)   # sorted, largest first
    return float(max(0.0, s[0] - s[1] - s[2] - s[3]))


def trace_distance(a, b):
    """0.5 * sum |eigenvalues of (a-b)| : how different two states are (0 to 1)."""
    d = (a - b + (a - b).conj().T) / 2
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(d))))


# ----------------------------------------------------------------------------
# 4. Open-system evolution on an isochore
# ----------------------------------------------------------------------------
def liouvillian(H, T, kappa, gamma):
    """16x16 matrix L such that d vec(rho)/dt = L vec(rho), with row-major vec.
    Rule used:  vec(A rho B) = kron(A, B^T) vec(rho).
      -i[H,rho]            -> -i (kron(H,I) - kron(I,H^T))
      kappa(rho_th - rho)  -> kappa (vec(rho_th) vec(I)^T - Identity)   (uses Tr rho = 1)
      gamma(Z rho Z - rho) -> gamma (kron(Z,Z^T) - Identity), for Z on each qubit."""
    I4 = np.eye(4, dtype=complex)
    I16 = np.eye(16, dtype=complex)
    L = -1j * (np.kron(H, I4) - np.kron(I4, H.T))
    rho_th = gibbs_state(H, T)
    L = L + kappa * (np.outer(rho_th.reshape(-1), I4.reshape(-1)) - I16)
    for S in (Z1, Z2):
        L = L + gamma * (np.kron(S, S.T) - I16)
    return L


def isochore_matrix(H, T, kappa, gamma, tau=TAU):
    """Evolution over time tau: vec(rho_after) = expm(L tau) vec(rho_before)."""
    return expm(liouvillian(H, T, kappa, gamma) * tau)


def apply_matrix(M, rho):
    return (M @ rho.reshape(-1)).reshape(4, 4)


# ----------------------------------------------------------------------------
# 5. The four-stage Otto cycle
# ----------------------------------------------------------------------------
def build_operators(p):
    """Everything that stays the same from cycle to cycle (computed once)."""
    problems = check_params(p)
    if problems:
        raise ValidationError(" ".join(problems))
    HH = hamiltonian(p["wH"], p["J"])
    HC = hamiltonian(p["wC"], p["J"])
    VH, EH = level_basis(p["wH"], p["J"])
    VC, EC = level_basis(p["wC"], p["J"])
    return {
        "HH": HH, "HC": HC, "VH": VH, "VC": VC, "EH": EH, "EC": EC,
        "PH": isochore_matrix(HH, p["TH"], p["kappa"], p["gamma"]),   # stage 1
        "PC": isochore_matrix(HC, p["TC"], p["kappa"], p["gamma"]),   # stage 3
        "A_exp": VC @ VH.T,     # stage 2: level k at hot -> level k at cold
        "A_comp": VH @ VC.T,    # stage 4: level k at cold -> level k at hot
    }


def one_cycle(ops, rho0):
    """Run stages 1-4 once, starting from rho0 (state at the start of the hot isochore).

    Sign convention: Q > 0 means heat flows INTO the qubits.
                     W > 0 means work done BY the qubits.
    Heat on an isochore (H constant):   Q = Tr[H (rho_after - rho_before)]
    Work on an adiabatic stroke:        W = sum_k p_k (E_k_before - E_k_after)
    First law over one cycle:           QH + QC - W = U_end - U_start
    """
    HH, HC = ops["HH"], ops["HC"]
    # Stage 1: hot isochore (H fixed = H(wH), thermalise with hot bath + decoherence)
    rho1 = apply_matrix(ops["PH"], rho0)
    U0, U1 = internal_energy(rho0, HH), internal_energy(rho1, HH)
    QH = U1 - U0
    # Stage 2: adiabatic expansion (no heat; populations of the levels are unchanged)
    rho2 = ops["A_exp"] @ rho1 @ ops["A_exp"].conj().T
    U2 = internal_energy(rho2, HC)
    pops1 = np.real(np.diag(ops["VH"].T @ rho1 @ ops["VH"]))
    W2 = float(np.sum(pops1 * (ops["EH"] - ops["EC"])))      # independent formula
    # Stage 3: cold isochore (H fixed = H(wC))
    rho3 = apply_matrix(ops["PC"], rho2)
    U3 = internal_energy(rho3, HC)
    QC = U3 - U2
    # Stage 4: adiabatic compression (back to H(wH))
    rho4 = ops["A_comp"] @ rho3 @ ops["A_comp"].conj().T
    U4 = internal_energy(rho4, HH)
    pops3 = np.real(np.diag(ops["VC"].T @ rho3 @ ops["VC"]))
    W4 = float(np.sum(pops3 * (ops["EC"] - ops["EH"])))
    W = W2 + W4
    # Energy-conservation check: compares heats (from isochores) with work (from level populations)
    eps = QH + QC - W - (U4 - U0)
    return {
        "QH": QH, "QC": QC, "W": W, "W2": W2, "W4": W4, "dU": U4 - U0, "eps": eps,
        "U": [U0, U1, U2, U3, U4], "rho": [rho0, rho1, rho2, rho3, rho4],
    }


def efficiency(QH, W):
    """eta = W/QH, only for a working heat engine (QH > 0 and W > 0); otherwise NaN."""
    if QH > 0 and W > 0:
        return W / QH
    return float("nan")


def carnot_efficiency(TH, TC):
    return 1.0 - TC / TH


def otto_efficiency(wH, wC):
    """Ideal Otto efficiency 1 - wC/wH of UNCOUPLED qubits (J = 0).  With J > 0 the efficiency
    can be below or above this number; only the Carnot bound is general."""
    return 1.0 - wC / wH


def engine_status(QH, QC, W):
    """Plain-English label for what the cycle is doing."""
    if QH > 0 and W > 0:
        return "Heat engine"
    if QH <= 0 and W > 0:
        return "Not a heat engine: produces work but takes no heat from the hot bath (it is using its starting energy)"
    if QH > 0 and W <= 0:
        return "Not a heat engine: takes heat but produces no net work (acts as a heater/refrigerator)"
    return "Not a heat engine: no heat taken from the hot bath and no net work"


def summarise_cycle(c, p):
    """Add efficiency, entanglement and status to a cycle result."""
    out = dict(c)
    out["eta"] = efficiency(c["QH"], c["W"])
    out["status"] = engine_status(c["QH"], c["QC"], c["W"])
    out["is_engine"] = bool(c["QH"] > 0 and c["W"] > 0)
    out["conc"] = [concurrence(r) for r in c["rho"]]
    out["clausius"] = -c["QH"] / p["TH"] - c["QC"] / p["TC"]   # >= 0 expected for two thermal baths
    return out


def simulate(p, rho0, max_cycles=MAX_CYCLES, n_forced=None):
    """Run the engine cycle after cycle starting from rho0.

    Returns a dict with
      'first'   : cycle 1 (starts from the prepared state rho0)
      'steady'  : the cycle after the state stops changing (the 'limit cycle')
      'works'   : work of each cycle that was run, list
      'n_cycles': number of cycles run, 'converged': True if the cycle closed
    If n_forced is given, exactly that many cycles are run (used for plots)."""
    check_state(rho0, "initial state")
    ops = build_operators(p)
    rho = rho0
    works, first, last = [], None, None
    limit = n_forced if n_forced is not None else max_cycles
    converged = False
    worst = {"hermitian_error": 0.0, "trace_error": 0.0, "min_eigenvalue": 0.0}
    max_eps = 0.0
    for n in range(1, limit + 1):
        c = one_cycle(ops, rho)
        works.append(c["W"])
        max_eps = max(max_eps, abs(c["eps"]))
        if n == 1:
            first = c
        for k, r in enumerate(c["rho"][1:], start=1):
            e = check_state(r, f"state after stage {k} of cycle {n}")
            worst["hermitian_error"] = max(worst["hermitian_error"], e["hermitian_error"])
            worst["trace_error"] = max(worst["trace_error"], e["trace_error"])
            worst["min_eigenvalue"] = min(worst["min_eigenvalue"], e["min_eigenvalue"])
        change = np.max(np.abs(c["rho"][4] - rho))
        rho, last = c["rho"][4], c
        if n_forced is None and change < TOL_STEADY:
            converged = True
            break
    if max_eps > TOL_ENERGY:
        raise ValidationError(f"Energy balance violated: error {max_eps:.2e} > {TOL_ENERGY:.0e}.")
    return {
        "first": summarise_cycle(first, p), "steady": summarise_cycle(last, p),
        "works": works, "n_cycles": len(works), "converged": converged,
        "max_energy_error": max_eps, "state_errors": worst, "params": dict(p),
    }


def memory_after_hot_isochore(p, rho_a, rho_b):
    """How much of the difference between two initial states survives stage 1?
    Returns D1/D0 where D = trace distance (1 = fully remembered, 0 = fully erased).
    For gamma = 0 this equals exp(-kappa*tau) exactly (checked in the tests)."""
    ops = build_operators(p)
    d0 = trace_distance(rho_a, rho_b)
    if d0 < 1e-14:
        return float("nan")
    d1 = trace_distance(apply_matrix(ops["PH"], rho_a), apply_matrix(ops["PH"], rho_b))
    return d1 / d0
