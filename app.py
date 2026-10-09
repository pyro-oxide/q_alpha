"""Q-Alpha: educational binary portfolio selection with QAOA.
Run: streamlit run app.py
"""
from __future__ import annotations

import io
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

st.set_page_config(page_title="Q-Alpha | Quantum Portfolio Lab", page_icon="⚛️", layout="wide")

EXAMPLE = pd.DataFrame({
    "Asset": ["AURORA", "BOREAL", "CIPHER", "DELTA", "ECHO"],
    "ExpectedReturn": [0.12, 0.10, 0.15, 0.08, 0.115],
    "Volatility": [0.20, 0.15, 0.30, 0.10, 0.18],
})


@dataclass
class PortfolioModel:
    names: list[str]
    mu: np.ndarray
    covariance: np.ndarray
    k: int
    risk_aversion: float
    penalty: float

    def __post_init__(self):
        self.n = len(self.names)
        # E(x) = constant + sum_i a_i x_i + sum_(i<j) b_ij x_i x_j
        self.constant = self.penalty * self.k**2
        self.linear = (self.risk_aversion * np.diag(self.covariance) / self.k**2
                       - self.mu / self.k + self.penalty * (1 - 2 * self.k))
        self.pairs = [(i, j,
                       self.risk_aversion * 2 * self.covariance[i, j] / self.k**2
                       + 2 * self.penalty)
                      for i in range(self.n) for j in range(i + 1, self.n)]
        self.bitstrings = np.array([[int((s >> i) & 1) for i in range(self.n)]
                                    for s in range(2**self.n)], dtype=int)
        self.energies = np.array([self.energy(x) for x in self.bitstrings])

    def energy(self, bits):
        x = np.asarray(bits, dtype=float)
        w = x / self.k
        return float(self.risk_aversion * (w @ self.covariance @ w)
                     - (self.mu @ w) + self.penalty * (x.sum() - self.k)**2)

    def metrics(self, bits):
        x = np.asarray(bits, dtype=float)
        w = x / self.k
        ret = float(self.mu @ w)
        var = float(w @ self.covariance @ w)
        return ret, np.sqrt(max(var, 0.0)), self.energy(bits)

    def describe(self, bits):
        return ", ".join(n for n, x in zip(self.names, bits) if x) or "None"


def make_circuit(model: PortfolioModel, angles: np.ndarray, measure=False):
    depth = len(angles) // 2
    qc = QuantumCircuit(model.n)
    qc.h(range(model.n))
    for layer in range(depth):
        gamma, beta = float(angles[layer]), float(angles[depth + layer])
        # A basis state |x> accumulates phase exp(-i gamma E(x));
        # the constant E contribution is a physically irrelevant global phase.
        for i, coeff in enumerate(model.linear):
            qc.p(-gamma * float(coeff), i)
        for i, j, coeff in model.pairs:
            qc.cp(-gamma * float(coeff), i, j)
        for i in range(model.n):
            qc.rx(2 * beta, i)
    if measure:
        qc.measure_all()
    return qc


def solve(model: PortfolioModel, depth: int, restarts: int, iterations: int,
          shots: int, seed: int):
    rng = np.random.default_rng(seed)
    timing_start = time.perf_counter()
    evaluations = [0]
    history = []

    def objective(theta):
        evaluations[0] += 1
        state = Statevector.from_instruction(make_circuit(model, theta))
        value = float(state.probabilities() @ model.energies)
        history.append(value)
        return value

    best = None
    for _ in range(restarts):
        x0 = np.r_[rng.uniform(0, 2 * np.pi, depth),
                   rng.uniform(0, np.pi, depth)]
        res = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": iterations, "rhobeg": 0.7})
        if best is None or res.fun < best.fun:
            best = res

    optimize_seconds = time.perf_counter() - timing_start
    bare = make_circuit(model, best.x)
    measured = make_circuit(model, best.x, measure=True)
    simulator = AerSimulator()
    sim_seed = int(seed) % (2**32 - 1)
    tc = transpile(measured, simulator, optimization_level=1, seed_transpiler=sim_seed)
    start = time.perf_counter()
    counts = simulator.run(tc, shots=shots, seed_simulator=sim_seed).result().get_counts()
    sampling_seconds = time.perf_counter() - start

    measured_indices = {int(b.replace(" ", ""), 2): int(v) for b, v in counts.items()}
    valid_counts = {i: c for i, c in measured_indices.items()
                    if model.bitstrings[i].sum() == model.k}
    choice = max(valid_counts, key=lambda i: (valid_counts[i], -model.energies[i])) if valid_counts else None

    classic_start = time.perf_counter()
    candidates = [i for i in range(2**model.n) if model.bitstrings[i].sum() == model.k]
    optimal_index = min(candidates, key=lambda i: model.energies[i])
    classical_seconds = time.perf_counter() - classic_start
    exact_probabilities = Statevector.from_instruction(bare).probabilities()

    return {
        "best_angles": best.x, "best_expectation": float(best.fun),
        "quantum_index": choice, "classical_index": optimal_index,
        "counts": measured_indices, "valid_counts": valid_counts,
        "exact_probabilities": exact_probabilities,
        "history": history, "evaluations": evaluations[0],
        "optimization_time": optimize_seconds, "sampling_time": sampling_seconds,
        "classical_time": classical_seconds, "circuit": bare, "measured": measured,
    }


def sample_data_csv():
    return EXAMPLE.to_csv(index=False).encode("utf-8")


st.title("⚛️ Q-Alpha | Quantum Portfolio Lab")
st.caption("Quantum Approximate Optimization Algorithm (QAOA) vs. exact classical search · For education, not financial advice")

with st.sidebar:
    st.header("Experiment settings")
    uploaded = st.file_uploader("Upload asset data (CSV)", type="csv")
    st.download_button("Download example CSV", sample_data_csv(), "example_assets.csv", "text/csv")
    st.caption("Required CSV columns: Asset, ExpectedReturn, Volatility. Returns and volatility must be decimals (e.g. 0.12 for 12%).")

try:
    data = pd.read_csv(uploaded) if uploaded is not None else EXAMPLE.copy()
    required = {"Asset", "ExpectedReturn", "Volatility"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(required - set(data.columns)))}")
    data = data[list(["Asset", "ExpectedReturn", "Volatility"])].copy()
    data["Asset"] = data["Asset"].astype(str).str.strip()
    data["ExpectedReturn"] = pd.to_numeric(data["ExpectedReturn"], errors="raise")
    data["Volatility"] = pd.to_numeric(data["Volatility"], errors="raise")
    if not 2 <= len(data) <= 7:
        raise ValueError("Use between 2 and 7 assets; large state spaces are not suitable for this demo.")
    if data.isna().any().any() or not np.isfinite(data[["ExpectedReturn", "Volatility"]].values).all():
        raise ValueError("Values must be finite and not missing.")
    if (data["Volatility"] <= 0).any():
        raise ValueError("All volatility values must be positive.")
    if data["Asset"].eq("").any() or data["Asset"].duplicated().any():
        raise ValueError("Asset names must be unique and nonempty.")
except Exception as exc:
    st.error(f"Invalid asset data: {exc}")
    st.stop()

n = len(data)
with st.sidebar:
    k = st.slider("Number of assets to select", 1, n, min(2, n))
    risk = st.slider("Risk aversion (λ)", 0.0, 20.0, 3.0, 0.25)
    rho = st.slider("Assumed common correlation", -0.1, 0.9, 0.2, 0.05,
                    help="Demo assumption: every pair of assets receives the same correlation. For real covariance modeling, use appropriate historical data.")
    penalty = st.slider("Cardinality constraint penalty", 0.5, 10.0, 3.0, 0.5)
    depth = st.slider("QAOA layers", 1, 3, 1)
    restarts = st.slider("Optimization restarts", 1, 8, 3)
    iterations = st.slider("Max iterations per restart", 30, 350, 120, 10)
    shots = st.select_slider("Measurement shots", options=[256, 512, 1024, 2048, 4096], value=1024)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run = st.button("▶ Run quantum experiment", type="primary", use_container_width=True)

mu = data["ExpectedReturn"].to_numpy(dtype=float)
vol = data["Volatility"].to_numpy(dtype=float)
correlation = np.full((n, n), rho)
np.fill_diagonal(correlation, 1.0)
cov = np.outer(vol, vol) * correlation
model = PortfolioModel(data["Asset"].tolist(), mu, cov, k, risk, penalty)

st.subheader("Asset assumptions")
st.dataframe(data.style.format({"ExpectedReturn": "{:.1%}", "Volatility": "{:.1%}"}),
             use_container_width=True, hide_index=True)
st.info("This is a toy **binary equal-weight selection** problem: exactly K assets are selected, each with portfolio weight 1/K. The covariance matrix assumes one shared pairwise correlation. No future returns are predicted.")

if not run:
    st.markdown("**Choose settings and click ‘Run quantum experiment’ to begin.**")
    st.stop()

with st.spinner("Optimizing QAOA parameters and sampling the quantum circuit..."):
    try:
        result = solve(model, depth, restarts, iterations, shots, int(seed))
    except Exception as exc:
        st.error(f"Quantum experiment failed: {exc}")
        st.stop()

opt = result["classical_index"]
q = result["quantum_index"]
classical_bits = model.bitstrings[opt]
best_return, best_vol, best_cost = model.metrics(classical_bits)
valid_shots = sum(result["valid_counts"].values())
optimal_shots = result["counts"].get(opt, 0)

st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Classical optimal objective", f"{best_cost:.5f}")
c2.metric("Valid quantum samples", f"{valid_shots / shots:.1%}")
c3.metric("Probability of exact optimum", f"{optimal_shots / shots:.1%}")
c4.metric("Quantum evaluations", str(result["evaluations"]))

col_q, col_c = st.columns(2)
with col_q:
    st.markdown("### Quantum (most frequent valid sample)")
    if q is None:
        st.warning("No valid selection appeared in the sampled shots. Try increasing shots, penalty, or optimization depth.")
    else:
        r, v, c = model.metrics(model.bitstrings[q])
        st.write("**Assets:**", model.describe(model.bitstrings[q]))
        st.write(f"**Expected return:** {r:.2%} · **Volatility:** {v:.2%}")
        st.write(f"**Objective:** {c:.5f}")
with col_c:
    st.markdown("### Exact classical optimum")
    st.write("**Assets:**", model.describe(classical_bits))
    st.write(f"**Expected return:** {best_return:.2%} · **Volatility:** {best_vol:.2%}")
    st.write(f"**Objective:** {best_cost:.5f}")

rows = []
for idx, count in result["counts"].items():
    bits = model.bitstrings[idx]
    ret, v, cost = model.metrics(bits)
    rows.append({"Portfolio": "".join(map(str, bits)), "Assets": model.describe(bits),
                 "Valid": bool(bits.sum() == k), "Shots": count, "Probability": count / shots,
                 "Expected return": ret, "Volatility": v, "Objective": cost,
                 "Optimal": bool(idx == opt)})
df = pd.DataFrame(rows).sort_values("Shots", ascending=False).reset_index(drop=True)

t1, t2, t3, t4 = st.tabs(["📊 Measurements", "📈 Risk and return", "⚙️ Circuit and training", "📤 Export"])
with t1:
    st.caption("Bitstrings are in asset order (left to right). Only selections containing exactly K assets satisfy the constraint.")
    chart_data = df.head(16).copy()
    # Treat portfolio bitstrings as text, not numbers
    chart_data["Portfolio"] = chart_data["Portfolio"].astype(str).str.zfill(n)
    # Create the bar chart
    fig = px.bar(
        chart_data,
        x="Portfolio",
        y="Shots",
        color="Valid",
        hover_data=["Assets", "Objective", "Optimal"],
        title="Measured portfolio outcomes"
    )
    # FIX: Display bitstrings as equally spaced categories
    fig.update_xaxes(
        type="category",
        title="Portfolio (binary selection)",
        categoryorder="array",
        categoryarray=chart_data["Portfolio"].tolist()
    )
    # Improve readability
    fig.update_layout(
        xaxis_tickangle=-45,
        bargap=0.2,
        yaxis_title="Measurement shots"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
with t2:
    all_rows = []
    for idx in itertools.combinations(range(n), k):
        bits = np.zeros(n, dtype=int)
        bits[list(idx)] = 1
        r, v, c = model.metrics(bits)
        all_rows.append({"Assets": model.describe(bits), "Risk (volatility)": v,
                         "Expected return": r, "Objective": c,
                         "Optimal": bool(np.array_equal(bits, classical_bits))})
    combos = pd.DataFrame(all_rows)
    fig = px.scatter(combos, x="Risk (volatility)", y="Expected return", color="Objective",
                     hover_name="Assets", symbol="Optimal", title="All feasible equal-weight portfolios")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower objective values are preferred. Return and volatility are annualized assumptions, not predictions.")
with t3:
    st.markdown("#### Gate-level QAOA circuit")
    st.code(str(result["circuit"].draw(output="text", fold=120)), language="text")
    st.markdown("#### Training objective by function evaluation")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=result["history"], mode="lines", name="Expected objective"))
    fig.update_layout(xaxis_title="Cost function evaluation", yaxis_title="Expected objective")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Optimized angles:", np.round(result["best_angles"], 4).tolist())
    st.write(f"QAOA expected objective: {result['best_expectation']:.6f}")
    st.write(f"Parameter optimization: {result['optimization_time']:.3f}s · Simulator shots: {result['sampling_time']:.3f}s · Classical enumeration: {result['classical_time']:.6f}s")
    st.caption("Classical enumeration is much cheaper for this tiny instance. QAOA is a hybrid quantum-classical heuristic, and these timings are not evidence of quantum advantage.")
with t4:
    st.download_button("Download measurement results (CSV)",
                       df.to_csv(index=False).encode("utf-8"), "qalpha_measurements.csv", "text/csv")
    st.download_button("Download circuit diagram (TXT)",
                       str(result["circuit"].draw(output="text", fold=120)), "qalpha_circuit.txt", "text/plain")
    st.download_button("Download portfolio comparison (CSV)",
                       combos.to_csv(index=False).encode("utf-8"), "qalpha_portfolios.csv", "text/csv")

st.divider()
st.markdown("""**How it works:** Each qubit represents whether an asset is selected. The cost-phase layer encodes a penalized risk/return objective via phase and controlled-phase gates. Mixer rotations explore alternative bitstrings. A classical optimizer adjusts circuit angles to reduce the expected objective, and an Aer simulator measures the final circuit. Exact enumeration provides a ground-truth reference.

**Hackathon deployment:** This dashboard executes locally with Qiskit Aer. For Qollab/IonQ eligibility, publish a separate runnable Qiskit circuit in Qollab using its supported simulator, and verify Qollab's available gate set and submission workflow. This app does not automatically upload or execute on Qollab.
""")
