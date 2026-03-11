"""W&B wrapper for autoresearch experiment tracking."""

import wandb

ENTITY = "oneiron-dev"
PROJECT = "ner-sft"

_run = None


def init(project: str = PROJECT, name: str = "", run_id: str = "",
         config: dict | None = None) -> wandb.sdk.wandb_run.Run:
    global _run
    _run = wandb.init(
        entity=ENTITY,
        project=project,
        id=run_id or None,
        name=name or run_id or None,
        resume="allow",
        config=config or {},
    )
    return _run


def log(metrics: dict, step: int | None = None) -> None:
    wandb.log(metrics, step=step)


def alert(title: str, text: str, level: str = "ERROR") -> None:
    lvl = getattr(wandb.AlertLevel, level, wandb.AlertLevel.ERROR)
    _run.alert(title=title, text=text, level=lvl)


def finish() -> None:
    global _run
    if _run:
        _run.finish()
        _run = None
