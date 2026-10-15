"""Small runtime capability probe; does not implement or train CTDA."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    result = {
        "purpose": "Runtime smoke test only; not a CTDA accuracy or resource benchmark",
        "timestamp_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "versions": {},
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": args.device,
        "checks": {},
    }
    for package in ["numpy", "torch", "scipy", "matplotlib", "PyYAML", "pandas", "scikit-learn", "tensorboard", "pytest"]:
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": props.name, "total_memory_mib": props.total_memory / 2**20}
        torch.cuda.reset_peak_memory_stats(device)

    # Exercise attention, layer norm, dropout, backward, and optimizer kernels.
    # This intentionally uses a small ordinary Transformer, not paper CTDA.
    layer = torch.nn.TransformerEncoderLayer(d_model=40, nhead=4,
                                            dim_feedforward=160, dropout=0.3,
                                            batch_first=True)
    model = torch.nn.Sequential(torch.nn.TransformerEncoder(layer, num_layers=2),
                                torch.nn.Linear(40, 40)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(8, 40, 40, device=device)
    target = torch.eye(40, device=device).expand(8, -1, -1)
    logits = model(x)
    p = torch.sigmoid(logits)
    pt = torch.where(target.bool(), p, 1 - p)
    alpha = torch.where(target.bool(), 0.75, 0.25)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (alpha * (1 - pt).square() * bce).mean()
    loss.backward()
    finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                           for parameter in model.parameters())
    before = next(model.parameters()).detach().clone()
    optimizer.step()
    updated = not torch.equal(before, next(model.parameters()).detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["gpu"]["peak_allocated_mib_small_probe_only"] = torch.cuda.max_memory_allocated(device) / 2**20
    result["checks"]["attention_forward_backward_adam"] = {
        "passed": bool(torch.isfinite(loss)) and finite_gradients and updated,
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": finite_gradients,
        "parameters_updated": updated,
        "shape": list(logits.shape),
    }

    costs = np.array([[1.0, 9.0], [8.0, 2.0]])
    rows, cols = linear_sum_assignment(costs)
    result["checks"]["numpy_scipy_assignment"] = {"passed": bool(np.array_equal(cols, [0, 1])),
                                                    "assignment_cost": float(costs[rows, cols].sum())}
    result["disk_free_gib"] = shutil.disk_usage(args.output.parent).free / 2**30
    result["all_checks_passed"] = all(check["passed"] for check in result["checks"].values())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
