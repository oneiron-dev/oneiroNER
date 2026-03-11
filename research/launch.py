"""Modal launcher for autoresearch NER experiments."""

import json
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

GPU_TYPE = os.getenv("AUTORESEARCH_GPU", "H100")
GPU_COSTS = {"H100": 3.95, "B200": 6.25, "A100": 3.30, "A10G": 1.10}

MODAL_TIMEOUT = 1080  # 18 min — 3 min buffer over internal 15 min cap

MODAL_APP_CODE = '''
import json
import modal
import os
import sys

app = modal.App("autoresearch-ner")

gpu_type = os.getenv("AUTORESEARCH_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0,<5.0.0",
        "accelerate>=0.28.0,<2.0.0",
        "huggingface_hub>=0.20.0,<2.0.0",
        "wandb",
        "numpy",
    )
    .add_local_dir("model", remote_path="/app/model", copy=True)
    .add_local_dir("research", remote_path="/app/research", copy=True)
    .add_local_dir("configs", remote_path="/app/configs", copy=True)
)

data_vol = modal.Volume.from_name("ner-data", create_if_missing=True)


@app.function(
    gpu=gpu_type,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    volumes={{"/data": data_vol}},
    timeout={timeout},
)
def train_experiment(experiment_json: str, run_id: str):
    import subprocess
    import os

    os.chdir("/app")
    os.environ["AUTORESEARCH_MODE"] = "1"
    os.environ["AUTORESEARCH_RUN_ID"] = run_id

    experiment = json.loads(experiment_json)

    cmd = [
        sys.executable, "-m", "model.train",
        "--lr", str(experiment["learning_rate"]),
        "--batch-size", str(experiment["per_device_train_batch_size"]),
        "--warmup-ratio", str(experiment.get("warmup_ratio", 0.05)),
        "--weight-decay", str(experiment.get("weight_decay", 0.01)),
        "--max-grad-norm", str(experiment.get("max_grad_norm", 1.0)),
        "--dropout", str(experiment.get("dropout", 0.1)),
        "--eval-steps", str(experiment.get("eval_steps", 100)),
        "--gold-ratio", str(experiment["gold_ratio"]),
        "--silver-en-ratio", str(experiment["silver_en_ratio"]),
        "--silver-ml-ratio", str(experiment["silver_ml_ratio"]),
        "--seed", str(experiment.get("seed", 42)),
        "--output-dir", "/data/checkpoints",
    ]

    max_steps = experiment.get("max_steps", -1)
    if max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])

    train_path = experiment.get("train_path", "/data/mini_train.jsonl")
    val_path = experiment.get("val_path", "/data/mini_eval.jsonl")
    cmd.extend(["--train-path", train_path, "--val-path", val_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {{
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }}


@app.local_entrypoint()
def main(experiment_json: str, run_id: str):
    result = train_experiment.remote(experiment_json, run_id)
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)
    if result["returncode"] != 0:
        sys.exit(result["returncode"])
'''


def generate_run_id(description: str = "baseline") -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = description.lower().replace(" ", "_")[:30]
    return f"ar_{desc}_{ts}"


def load_experiment() -> dict:
    from research.train import EXPERIMENT
    experiment = dict(EXPERIMENT)
    experiment["eval_steps"] = min(experiment.get("eval_steps", 100), 100)
    return experiment


def estimate_cost(experiment: dict) -> float:
    gpu = os.getenv("AUTORESEARCH_GPU", "H100")
    cost_per_hr = GPU_COSTS.get(gpu, 3.95)
    est_minutes = experiment.get("max_wallclock_minutes", 15) + 3
    return cost_per_hr * (est_minutes / 60)


def launch_modal(experiment: dict, run_id: str) -> dict:
    experiment_json = json.dumps(experiment)

    modal_script = MODAL_APP_CODE.format(timeout=MODAL_TIMEOUT)

    script_path = "/tmp/autoresearch_modal_app.py"
    with open(script_path, "w") as f:
        f.write(modal_script)

    import shutil
    modal_bin = shutil.which("modal")
    if modal_bin is None:
        raise RuntimeError("modal CLI not found in PATH")

    cmd = [
        modal_bin, "run", script_path,
        "--experiment-json", experiment_json,
        "--run-id", run_id,
    ]

    logger.info("Launching Modal job: gpu=%s, run_id=%s", GPU_TYPE, run_id)
    logger.info("Estimated cost: $%.2f", estimate_cost(experiment))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    proc.wait()
    output = "".join(output_lines)

    metrics = None
    for line in output_lines:
        if line.strip().startswith("AUTORESEARCH_METRICS:"):
            metrics_json = line.strip().split("AUTORESEARCH_METRICS:", 1)[1]
            metrics = json.loads(metrics_json)

    return {
        "exit_code": proc.returncode,
        "output": output,
        "metrics": metrics,
        "run_id": run_id,
    }


def poll_wandb(run_id: str, poll_interval: int = 30, max_wait: int = 1200) -> dict:
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        logger.warning("wandb not installed — skipping W&B polling")
        return {}

    project = "oneiron-dev/ner-sft"
    start = time.time()

    while time.time() - start < max_wait:
        try:
            api.flush()
            run = api.run(f"{project}/{run_id}")

            if run.state in ("crashed", "failed"):
                return {
                    "state": run.state,
                    "smoke_abort": run.summary.get("smoke_abort", False),
                    "summary": dict(run.summary),
                }

            smoke_abort = run.summary.get("smoke_abort", False)
            if smoke_abort:
                return {
                    "state": run.state,
                    "smoke_abort": True,
                    "abort_reason": run.summary.get("abort_reason", ""),
                    "summary": dict(run.summary),
                }

            if run.state == "finished":
                return {
                    "state": "finished",
                    "smoke_abort": False,
                    "summary": dict(run.summary),
                }
        except Exception as e:
            logger.debug("W&B poll error: %s", e)

        time.sleep(poll_interval)

    return {"state": "timeout", "smoke_abort": False}


def append_results_tsv(commit: str, metrics: dict, status: str,
                       description: str, path: str = "results.tsv"):
    import fcntl
    from research.score import format_tsv_row, TSV_HEADER

    row = format_tsv_row(commit, metrics, status, description)

    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if os.path.getsize(path) == 0:
                f.write(TSV_HEADER + "\n")
            f.write(row + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    logger.info("Appended result: %s [%s]", commit[:8], status)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    experiment = load_experiment()
    run_id = generate_run_id()

    logger.info("=== Autoresearch Launch ===")
    logger.info("Run ID: %s", run_id)
    logger.info("GPU: %s ($%.2f/hr)", GPU_TYPE, GPU_COSTS.get(GPU_TYPE, 0))
    logger.info("Token budget: %d", experiment.get("token_budget", 0))
    logger.info("Eval steps: %d", experiment.get("eval_steps", 100))

    result = launch_modal(experiment, run_id)

    if result["metrics"]:
        logger.info("=== Metrics ===")
        for k, v in sorted(result["metrics"].items()):
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %s", k, v)
    else:
        logger.warning("No AUTORESEARCH_METRICS found in output")

    logger.info("Exit code: %d", result["exit_code"])


if __name__ == "__main__":
    main()
