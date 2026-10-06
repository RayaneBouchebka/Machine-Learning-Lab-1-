import time
import os
import psutil
import joblib
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Set Seeds for Reproducibility
random.seed(42)
np.random.seed(42)

# Load & Split Data (70/30 stratified)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracies = {}
process = psutil.Process(os.getpid())

print("--- Starting Benchmarks ---")

for name, model in models.items():
    # Warm-up run
    model.fit(X_train, y_train)
    
    # Training Time (5 runs, median)
    train_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        m = clone(model)
        m.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    median_train_time = np.median(train_times)

    # Test Accuracy
    acc = model.score(X_test, y_test)
    accuracies[name] = round(acc, 4)

    # Single-sample Inference Latency (1 warm-up, 100 runs, median)
    sample = X_test[0:1]
    model.predict(sample)
    
    inf_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        inf_times.append((time.perf_counter() - t0) * 1000)
    median_inf_latency = np.median(inf_times)

    # Save Model & File Size
    filepath = f"lab01/results/{name}.joblib"
    joblib.dump(model, filepath)
    size_bytes = os.path.getsize(filepath)
    size_kb = size_bytes / 1024.0

    # Peak RSS Memory
    mem_mb = process.memory_info().rss / (1024 * 1024)

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Median Training Time: {median_train_time:.4f} s")
    print(f"Median Single-Sample Latency: {median_inf_latency:.4f} ms")
    print(f"Model Size: {size_kb:.2f} KB ({size_bytes} bytes)")
    print(f"Peak Process Memory: {mem_mb:.2f} MB")

df_acc = pd.DataFrame(list(accuracies.items()), columns=["model", "test_accuracy"])
df_acc.to_csv("lab01/results/baseline_accuracy.csv", index=False)
print("\nSaved accuracy to lab01/results/baseline_accuracy.csv")
