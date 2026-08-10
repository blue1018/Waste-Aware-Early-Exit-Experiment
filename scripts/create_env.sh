#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="waste-early-exit"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${PROJECT_ROOT}/environment.yml" --prune
else
  conda env create --file "${PROJECT_ROOT}/environment.yml"
fi

conda run --name "${ENV_NAME}" python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (waste-early-exit)"
conda run --name "${ENV_NAME}" python -c '
import torch

print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available. Run this script in a native macOS terminal.")

device = torch.device("mps")
model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).to(device)
x = torch.randn(3, 8, device=device)
loss = model(x).square().mean()
loss.backward()
torch.mps.synchronize()
print("MPS forward/backward check: passed")
'

echo "Environment ready. Run: conda activate ${ENV_NAME}"
