"""Train the two members of Giano on the official Trentino window protocol."""

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from filelock import FileLock

from .model import INPUTS, VARIABLES, LongGapModel
from .physical import physical_prediction
from .trentino import PROTOCOL


class Windows:
    def __init__(self, folder: Path):
        self.fields = {
            key: np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in (*INPUTS, "target", "target_mask", "target_physical")
        }
        self.legacy = np.load(folder / "legacy.npy", allow_pickle=False)

    def batch(self, indices):
        return {
            key: torch.from_numpy(array[indices]) for key, array in self.fields.items()
        }


def permutation(size, seed, count):
    result = []
    while len(result) < count:
        generator = torch.Generator().manual_seed(
            seed + 1000003 * (len(result) // size)
        )
        result.extend(torch.randperm(size, generator=generator).tolist())
    return np.array(result[:count])


def training_order(legacy, seed, rounds):
    old = np.flatnonzero(legacy)
    additional = np.flatnonzero(~legacy)
    balanced = np.concatenate(
        [old[permutation(len(old), seed + 7000001, len(additional))], additional]
    )
    return balanced[permutation(len(balanced), seed, rounds * 1860)].reshape(
        rounds, 1860
    )


def training_device(requested: str) -> torch.device:
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested in ("auto", "cpu"):
        return torch.device("cpu")
    raise ValueError(
        f"Training device {requested!r} is unavailable in this PyTorch environment"
    )


def model_inputs(batch, device):
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda") for key in INPUTS
    }


def cpu_snapshot(model):
    return {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()}


def train_batch(model, optimizer, batch, device, *, bf16=False):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
        prediction, _ = model(**model_inputs(batch, device))
    if device.type == "cpu":
        # Preserve the reduction order of the published CPU run.
        losses = []
        for j in range(4):
            mask = batch["target_mask"][..., j]
            if mask.any():
                losses.append(
                    (prediction[:, :, 0, j][mask] - batch["target"][..., j][mask])
                    .abs()
                    .mean()
                )
        loss = torch.stack(losses).mean()
    else:
        mask = batch["target_mask"][..., :4]
        counts = mask.sum(dim=(0, 1))
        n_variables = int((counts > 0).sum())
        if not n_variables:
            raise ValueError("Training batch has no supervised targets")
        target = batch["target"][..., :4].to(device)
        errors = torch.where(
            mask.to(device), prediction[:, :, 0, :4].float() - target, 0.0
        )
        loss = (
            errors.abs().sum(dim=(0, 1)) / counts.clamp_min(1).to(device)
        ).sum() / n_variables
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


@torch.inference_mode()
def development_score(model, windows, *, preview, device):
    """Mean physical MAE divided by training scale over four variables and both rosters."""
    values = []
    for ids in (np.arange(len(windows.legacy)), np.flatnonzero(windows.legacy)):
        errors, counts = np.zeros(4), np.zeros(4, dtype=int)
        for start in range(0, len(ids), 8):
            batch = windows.batch(ids[start : start + 8])
            pred, _ = model(**model_inputs(batch, device))
            # Keep physical metrics in float64 on the host, including on Metal.
            pred = pred[:, :, :1, :4].cpu()
            for j, var in enumerate(VARIABLES[:4]):
                mask = batch["target_mask"][..., j]
                scale = batch["scale"][:, :1, j]
                physical = physical_prediction(
                    pred[:, :, :1, j], batch["center"][:, :1, j], scale, variable=var
                )[..., 0].double()
                absolute = (
                    physical - batch["target_physical"][..., j].double()
                ).abs() / scale.double()
                errors[j] += float(absolute[mask].sum())
                counts[j] += int(mask.sum())
        if not preview and not counts.all():
            raise ValueError("A development variable has no measured targets")
        values.extend((errors[counts > 0] / counts[counts > 0]).tolist())
    return math.fsum(values) / len(values)


def save_state(payload, target):
    with TemporaryDirectory(dir=target.parent) as tmp:
        staged = Path(tmp) / "state.pt"
        torch.save(payload, staged)
        os.replace(staged, target)


def train(
    cache: Path,
    output: Path,
    *,
    fixed_rounds=False,
    resume=False,
    device="auto",
    precision="auto",
):
    device = training_device(device)
    native_bf16 = (
        device.type == "cuda"
        and torch.cuda.get_device_capability()[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    if precision not in ("auto", "fp32", "bf16") or (
        precision == "bf16" and not native_bf16
    ):
        raise ValueError(
            "Choose fp32 or auto on this device. bf16 requires CUDA with native bfloat16 support."
        )
    bf16 = native_bf16 and precision != "fp32"
    runtime = {"device": device.type, "precision": "bf16" if bf16 else "fp32"}
    print(f"Training on {device} ({runtime['precision']})", flush=True)
    recipe = json.loads((cache / "recipe.json").read_text())
    if recipe["protocol"] != 1:
        raise ValueError("Unknown training cache format")
    preview = recipe["preview"]
    train_set, validation = Windows(cache / "train"), Windows(cache / "val")
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=resume)
    with FileLock(str(output / "train.lock"), timeout=0):
        initial = torch.load(
            PROTOCOL / "initial.pt", map_location="cpu", weights_only=True
        )
        curves = json.loads((PROTOCOL / "learning-curves.json").read_text())
        ensemble, report = [], {}
        for index, seed in enumerate((42, 43)):
            selected = curves[str(seed)]["selected_round"]
            rounds = 1 if preview else selected if fixed_rounds else 36
            order = (
                np.arange(len(train_set.legacy))[None]
                if preview
                else training_order(train_set.legacy, seed, rounds)
            )
            model = LongGapModel().eval().to(device)
            model.load_state_dict(initial[index])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                fused=True if device.type == "cuda" else None,
            )
            # A continuation belongs to the same cache and sampling schedule.
            identity = (seed, str(cache.resolve()), preview, fixed_rounds, rounds)
            checkpoint = output / f"seed-{seed}.pt"
            history = []
            best = cpu_snapshot(model)
            best_score = (
                math.inf
                if fixed_rounds or preview or (resume and checkpoint.exists())
                else development_score(model, validation, preview=False, device=device)
            )
            best_round = 0
            if resume and checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved["identity"] != identity:
                    raise ValueError("Resume requires the same cache and training mode")
                previous = saved.get("runtime", {"device": "cpu", "precision": "fp32"})
                if previous != runtime:
                    raise ValueError(
                        f"Resume with --device {previous['device']} --precision {previous['precision']}"
                    )
                model.load_state_dict(saved["last"])
                optimizer.load_state_dict(saved["optimizer"])
                history, best, best_score, best_round = (
                    saved[k] for k in ("history", "best", "score", "round")
                )
            for epoch in range(len(history), rounds):
                samples = order[epoch]
                losses = [
                    train_batch(
                        model,
                        optimizer,
                        train_set.batch(samples[i : i + 8]),
                        device,
                        bf16=bf16,
                    )
                    for i in range(0, len(samples), 8)
                ]
                score = None
                if not fixed_rounds or epoch + 1 == rounds:
                    score = development_score(
                        model, validation, preview=preview, device=device
                    )
                    if score < best_score:
                        best_score, best_round = score, epoch + 1
                        best = cpu_snapshot(model)
                history.append(
                    {
                        "round": epoch + 1,
                        "loss": math.fsum(torch.stack(losses).cpu().tolist())
                        / len(losses),
                        "development": score,
                    }
                )
                save_state(
                    dict(
                        identity=identity,
                        runtime=runtime,
                        history=history,
                        best=best,
                        score=best_score,
                        round=best_round,
                        last=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                    ),
                    checkpoint,
                )
                print(
                    f"seed {seed}, round {epoch + 1}/{rounds}, loss {history[-1]['loss']:.6f}, development {score}",
                    flush=True,
                )
            ensemble.append(best)
            report[str(seed)] = {"selected_round": best_round, "history": history}
        if not preview:
            released = torch.load(
                Path(__file__).with_name("weights") / "ensemble.pt",
                map_location="cpu",
                weights_only=True,
            )
            for seed, fitted, public in zip((42, 43), ensemble, released, strict=True):
                difference = max(
                    float((fitted[k] - public[k]).abs().max()) for k in public
                )
                report[str(seed)]["max_parameter_difference_from_release"] = difference
                print(
                    f"Seed {seed}, maximum parameter difference from release: {difference:.8g}"
                )
        save_state(ensemble, output / "ensemble.pt")
        with (output / "ensemble.pt").open("rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        (output / "training.json").write_text(
            json.dumps(
                {
                    **runtime,
                    "preview": preview,
                    "fixed_rounds": fixed_rounds,
                    "ensemble_sha256": checksum,
                    "seeds": report,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Use CUDA, then Apple Metal, then CPU when selected automatically",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="Use mixed bfloat16 on supported CUDA GPUs and float32 elsewhere",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fixed-rounds",
        action="store_true",
        help="Replay the published stopping rounds (32 and 20)",
    )
    args = parser.parse_args()
    train(
        args.cache,
        args.output,
        fixed_rounds=args.fixed_rounds,
        resume=args.resume,
        device=args.device,
        precision=args.precision,
    )
