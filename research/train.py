"""Autoresearch experiment config — the ONLY file the agent edits."""

EXPERIMENT = {
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "per_device_train_batch_size": 32,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gold_ratio": 0.75,
    "silver_en_ratio": 0.20,
    "silver_ml_ratio": 0.05,
    "dropout": 0.1,
    "seed": 42,
    "token_budget": 3_200_000,
    "max_wallclock_minutes": 15,
    "eval_steps": 100,
    "train_path": "/data/train.jsonl",
    "val_path": "/data/val.jsonl",
}

MODEL_OVERRIDES = {
    # Phase 2 knobs — do not enable until Phase 1 converges
    # "classifier_hidden": None,
    # "classifier_dropout": 0.1,
    # "class_weights": None,
    # "focal_loss_gamma": None,
    # "encoder_freeze_layers": 0,
    # "label_smoothing": 0.0,
}
