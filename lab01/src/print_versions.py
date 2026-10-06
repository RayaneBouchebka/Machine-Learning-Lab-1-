import sys
import numpy, pandas, sklearn, scipy, matplotlib, seaborn, torch, torchvision, torchinfo, thop, onnx, onnxruntime, mlflow, memory_profiler, psutil, codecarbon, fastapi, uvicorn, pytest, httpx, locust, requests, pyarrow, joblib, tqdm

print(f"Python version: {sys.version}")
pkgs = [numpy, pandas, sklearn, scipy, matplotlib, seaborn, torch, torchvision, torchinfo, thop, onnx, onnxruntime, mlflow, memory_profiler, psutil, codecarbon, fastapi, uvicorn, pytest, httpx, locust, requests, pyarrow, joblib, tqdm]

for p in pkgs:
    ver = getattr(p, "__version__", "N/A")
    print(f"{p.__name__}: {ver}")
