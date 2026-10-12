"""Modal app for running the heavy Nastaliq jobs on cloud GPUs.

Local Apple-MPS training is fine for iteration but throttles and serialises
long runs. This app runs the expensive/repetitive jobs (combined flow +
recognition-in-the-loop, sweeps, bigger recogniser) on a Modal GPU.

Layout inside the container:
    /root/repo                 code (src/, experiments/ baked into the image)
    /root/repo/data            datasets + manifests      (Volume: nastaliq-data)
    /root/repo/checkpoints     model checkpoints/outputs (Volume: nastaliq-checkpoints)

The manifests store paths relative to the repo root (``data/clean/...``), so
the data volume is mounted *at* ``/root/repo/data`` with its contents at the
volume root.

Usage:
    # one-time: upload data + checkpoints
    modal volume create nastaliq-data
    modal volume put -rf nastaliq-data data/clean /clean
    modal volume put -rf nastaliq-data data/damaged /damaged
    modal volume put -rf nastaliq-data data/damage_masks /damage_masks
    modal volume put -f nastaliq-data data/manifest.json /manifest.json
    modal volume put -f nastaliq-data data/manifest_train.json /manifest_train.json
    modal volume put -f nastaliq-data data/manifest_val.json /manifest_val.json
    modal volume create nastaliq-checkpoints
    modal volume put -f nastaliq-checkpoints checkpoints/baseline_best.pt /baseline_best.pt

    # smoke test
    modal run modal_app.py --task smoke

    # a real run
    modal run modal_app.py --task restoration --epochs 20
    modal run modal_app.py --extra "experiments/eval_restore_ocr.py --device cuda --restoration-checkpoint /root/repo/checkpoints/xxx.pt --out /root/repo/checkpoints/eval.json"
"""

import modal

app = modal.App("nastaliq-restore")
GPU = "L4"  # plenty for these ~2M-param models; swap to A100 for anything heavy

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchvision", "opencv-python-headless",
        "scikit-image", "matplotlib", "pillow", "numpy",
    )
    .add_local_dir("src", remote_path="/root/repo/src")
    .add_local_dir("experiments", remote_path="/root/repo/experiments")
)

data_vol = modal.Volume.from_name("nastaliq-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("nastaliq-checkpoints", create_if_missing=True)

PRESETS = {
    "smoke": [
        "src/train_restoration.py", "--device", "cuda", "--epochs", "1",
        "--limit-train", "32", "--limit-val", "16",
        "--checkpoint", "/root/repo/checkpoints/smoke.pt",
    ],
    "restoration": [
        "src/train_restoration.py", "--device", "cuda", "--epochs", "20",
        "--checkpoint", "/root/repo/checkpoints/restoration_a100.pt",
        "--history-out", "/root/repo/checkpoints/restoration_a100_history.json",
    ],
    "flow": [
        "src/train_flow.py", "--device", "cuda", "--epochs", "20",
        "--sample-steps", "12",
        "--checkpoint", "/root/repo/checkpoints/restoration_flow_a100.pt",
        "--history-out", "/root/repo/checkpoints/restoration_flow_a100_history.json",
    ],
    # flow matching + recognition-in-the-loop (one-step reconstruction)
    "flow-ocr": [
        "src/train_flow.py", "--device", "cuda", "--epochs", "20",
        "--sample-steps", "12", "--lambda-ocr", "0.1", "--num-workers", "4",
        "--val-limit", "265",
        "--checkpoint", "/root/repo/checkpoints/restoration_flow_ocr.pt",
        "--history-out", "/root/repo/checkpoints/restoration_flow_ocr_history.json",
    ],
}

# Heavier jobs (backprop through a sampling loop) go on an A100.
HEAVY_TASKS = {"flow-ocr"}


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/repo/data": data_vol, "/root/repo/checkpoints": ckpt_vol},
    timeout=60 * 60 * 4,
)
def run_remote(args: list):
    import os
    import subprocess
    import sys

    print("RUN:", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd="/root/repo", check=True,
                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
    data_vol.commit()
    ckpt_vol.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/root/repo/data": data_vol, "/root/repo/checkpoints": ckpt_vol},
    timeout=60 * 60 * 4,
)
def run_remote_a100(args: list):
    import os
    import subprocess
    import sys

    print("RUN (A100):", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd="/root/repo", check=True,
                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
    data_vol.commit()
    ckpt_vol.commit()


@app.local_entrypoint()
def main(task: str = "smoke", epochs: int = 20, extra: str = ""):
    if extra:
        args = extra.split()
    else:
        args = list(PRESETS.get(task, []))
        if task in ("restoration", "flow", "flow-ocr"):
            # replace the hard-coded epoch count with --epochs
            args = [str(epochs) if a == "20" else a for a in args]
    if not args:
        raise SystemExit(f"unknown task '{task}'; pass --extra '...'")
    runner = run_remote_a100 if task in HEAVY_TASKS else run_remote
    runner.remote(args)


# ---------------------------------------------------------------------------
# Sweep: multi-seed error bars + lambda_ocr tuning (runs in parallel on Modal)
# ---------------------------------------------------------------------------

SWEEP_SEEDS = [0, 1, 2]
SWEEP_LAMBDAS = ["0.03", "0.3", "1.0"]  # 0.1 is covered by the seed grid
SWEEP_EPOCHS = 20


def _stub(name):
    return f"/root/repo/checkpoints/{name}"


def _train_eval(train_args, ckpt, eval_extra=None):
    return {
        "train": train_args,
        "eval": (["experiments/eval_restore_ocr.py", "--device", "cuda",
                  "--restoration-checkpoint", ckpt] + (eval_extra or []) +
                 ["--out", ckpt[:-3] + "_eval.json"]),
    }


def _sweep_jobs():
    jobs = []
    for s in SWEEP_SEEDS:
        # U-Net baseline
        ck = _stub(f"sw_baseline_s{s}.pt")
        jobs.append(_train_eval(
            ["src/train_restoration.py", "--device", "cuda", "--epochs", str(SWEEP_EPOCHS),
             "--seed", str(s), "--num-workers", "4", "--checkpoint", ck,
             "--history-out", ck[:-3] + "_hist.json"], ck))
        # U-Net + RITL (lambda 0.1)
        ck = _stub(f"sw_ritl_s{s}.pt")
        jobs.append(_train_eval(
            ["src/train_restoration.py", "--device", "cuda", "--epochs", str(SWEEP_EPOCHS),
             "--seed", str(s), "--num-workers", "4", "--lambda-ocr", "0.1", "--checkpoint", ck,
             "--history-out", ck[:-3] + "_hist.json"], ck))
        # flow
        ck = _stub(f"sw_flow_s{s}.pt")
        jobs.append(_train_eval(
            ["src/train_flow.py", "--device", "cuda", "--epochs", str(SWEEP_EPOCHS),
             "--seed", str(s), "--num-workers", "4", "--checkpoint", ck,
             "--history-out", ck[:-3] + "_hist.json"], ck, ["--flow-steps", "12"]))
        # flow + RITL (lambda 0.1)
        ck = _stub(f"sw_flowritl_s{s}.pt")
        jobs.append(_train_eval(
            ["src/train_flow.py", "--device", "cuda", "--epochs", str(SWEEP_EPOCHS),
             "--seed", str(s), "--num-workers", "4", "--lambda-ocr", "0.1", "--checkpoint", ck,
             "--history-out", ck[:-3] + "_hist.json"], ck, ["--flow-steps", "12"]))
    # lambda_ocr sweep at seed 0
    for lam in SWEEP_LAMBDAS:
        tag = lam.replace(".", "p")
        ck = _stub(f"sw_ritl_l{tag}_s0.pt")
        jobs.append(_train_eval(
            ["src/train_restoration.py", "--device", "cuda", "--epochs", str(SWEEP_EPOCHS),
             "--seed", "0", "--num-workers", "4", "--lambda-ocr", lam, "--checkpoint", ck,
             "--history-out", ck[:-3] + "_hist.json"], ck))
    return jobs


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/repo/data": data_vol, "/root/repo/checkpoints": ckpt_vol},
    timeout=60 * 60 * 3,
)
def train_and_eval(train_args: list, eval_args: list):
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print("TRAIN:", " ".join(train_args), flush=True)
    subprocess.run([sys.executable, *train_args], cwd="/root/repo", check=True, env=env)
    print("EVAL:", " ".join(eval_args), flush=True)
    subprocess.run([sys.executable, *eval_args], cwd="/root/repo", check=True, env=env)
    data_vol.commit()
    ckpt_vol.commit()
    return eval_args[eval_args.index("--out") + 1]


@app.local_entrypoint()
def sweep():
    jobs = _sweep_jobs()
    print(f"launching {len(jobs)} train+eval jobs on {GPU}...", flush=True)
    outs = list(train_and_eval.starmap([(j["train"], j["eval"]) for j in jobs]))
    print(f"completed {len(outs)} jobs:")
    for o in outs:
        print("  ", o)
