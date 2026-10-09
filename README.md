# Q-Alpha — Quantum Portfolio Lab

Educational QAOA portfolio-selection web application, written entirely in Python.

## Run

Requires Python 3.10+ (3.11 or 3.12 recommended).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app launches in a local browser. Click **Run quantum experiment**.

## Included

- Upload up to 7 assets via CSV (Asset, ExpectedReturn, Volatility)
- Adjust selection count, risk aversion, correlation, penalty and circuit depth
- Gate-based QAOA: phase (P), controlled phase (CP), and RX mixer gates
- SciPy classical parameter optimization with multiple restarts
- Qiskit Aer circuit execution with actual sampled measurements
- Exact classical optimum via exhaustive enumeration
- Sampling histograms, risk/return scatterplot, circuit diagram, training history, CSV export

## Mathematical model

Let `x_i` be 0/1, `k` the number of selected assets, `mu` assumed annual returns, and `Sigma` covariance:

`E(x) = lambda * (x/k)^T Sigma (x/k) - mu^T (x/k) + P * (sum(x)-k)^2`

The penalty makes invalid asset counts costly. The risk/return component is calculated for equal weights `1/k` (the same definition is used in graphs and reported metrics). Binary quadratic terms are translated directly to `p` and `cp` gates. QAOA parameters are optimized using a local statevector; **final outcome frequencies** are measured using Qiskit Aer simulator shots.

## Limitations

- Toy educational asset data, not trading or investment advice.
- Shared pairwise correlation is an illustrative assumption, not fitted to real markets.
- Small problem sizes only. Exact classical enumeration often wins on speed and quality.
- Quantum output is approximate and can fail to sample a feasible portfolio.
- **Qollab/IonQ deployment is not automatic.** Port the `make_circuit` function to their supported backend, check transpilation, and publish/test a runnable circuit there.
- No claim of quantum financial or computational advantage.
