# Lab 01 Report: Environment and First System Measurements

## 1. Goal
Set up a reproducible isolated Python environment, train baseline linear and tree ensemble models on the Breast Cancer Wisconsin dataset, and evaluate their deployment feasibility against system resource constraints (Cloud, Edge, Mobile, and TinyML).

## 2. Method
- Created an isolated `.venv` environment and installed pinned dependencies.
- Loaded the Breast Cancer Wisconsin dataset using `sklearn.datasets.load_breast_cancer`.
- Applied a 70/30 stratified train/test split with `random_state=42`.
- Trained two baseline models:
  - `LogisticRegression(max_iter=1000, random_state=42)` wrapped in a `StandardScaler` pipeline.
  - `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Recorded test accuracy in `lab01/results/baseline_accuracy.csv`.
- Measured system resource usage:
  - Training time: median of 5 iterations following 1 warm-up run using `sklearn.base.clone`.
  - Single-sample inference latency: median of 100 predictions.
  - Model file size: saved using `joblib.dump` and measured in bytes/KB.
  - Peak memory: process RSS measured via `psutil`.

## 3. Results

### Baseline Accuracy & System Metrics
| Model | Test Accuracy | Median Train Time (s) | Single-Sample Latency (ms) | Model Size (KB) | Peak RSS Memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LogisticRegression** | 0.9883 | 0.0044 | 0.0649 | 2.23 | 142.72 |
| **RandomForestClassifier** | 0.9357 | 0.0975 | 1.3721 | 284.07 | 144.41 |

### Platform Resource Budget Fit
| Target Platform | Memory Budget | Latency Budget | Model Size Budget | Logistic Regression | Random Forest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cloud** | >= 1 GB | <= 100 ms | <= 500 MB | Suitable | Suitable |
| **Edge** | 256–1024 MB | <= 50 ms | <= 50 MB | Suitable | Suitable |
| **Mobile** | 64–256 MB | <= 20 ms | <= 10 MB | Suitable | Suitable |
| **TinyML** | <= 256 KB | <= 10 ms | <= 100 KB | Unsuitable (Memory) | Unsuitable (Size & Memory) |

## 4. Conclusions
1. Scaling features is critical for linear models; applying standard scaling eliminated convergence warnings and yielded higher classification accuracy (98.83%) compared to the tree ensemble baseline (93.57%).
2. Simple linear models offer substantial efficiency advantages for edge and mobile deployments, delivering roughly 21x lower single-sample inference latency and 127x smaller disk footprints compared to Random Forest ensembles.
3. Standard Python execution environments cannot target TinyML constraints without C/C++ model export or micro-framework quantization, as Python runtime process memory (~142–144 MB RSS) far exceeds TinyML memory budgets (<= 256 KB).
