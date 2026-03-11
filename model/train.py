"""NER baseline training with HF Trainer + mixture sampling."""

import argparse
import json
import logging
import math
import os

import torch
from torch.utils.data import Sampler
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from model.config import (
    MODEL_NAME, NUM_LABELS, ID2LABEL,
    GOLD_RATIO, SILVER_EN_RATIO, SILVER_ML_RATIO,
    LEARNING_RATE, WARMUP_RATIO, BATCH_SIZE, NUM_EPOCHS,
    WEIGHT_DECAY, MAX_GRAD_NORM, load_type_mapping,
)
from model.ner_model import NerModel
from model.ner_dataset import NerDataset, ner_collate_fn
from model.eval import compute_metrics_for_trainer

logger = logging.getLogger(__name__)


class MixtureSampler(Sampler):
    """Samples from gold/silver_en/silver_ml buckets at target ratios."""

    def __init__(self, dataset, gold_ratio=GOLD_RATIO, silver_en_ratio=SILVER_EN_RATIO, silver_ml_ratio=SILVER_ML_RATIO, seed=42):
        self.buckets = {"gold": [], "silver_en": [], "silver_ml": []}
        for i, rec in enumerate(dataset.records):
            bucket = rec.get("bucket", "gold")
            if bucket in self.buckets:
                self.buckets[bucket].append(i)
            else:
                self.buckets["gold"].append(i)

        self.ratios = {"gold": gold_ratio, "silver_en": silver_en_ratio, "silver_ml": silver_ml_ratio}
        self.seed = seed
        self.epoch = 0

        # Total length = full dataset size; each bucket contributes its ratio share
        self._len = len(dataset)

        for name, indices in self.buckets.items():
            logger.info("MixtureSampler bucket %s: %d records (target %.0f%%)",
                        name, len(indices), self.ratios[name] * 100)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for bucket_name, ratio in self.ratios.items():
            bucket_indices = self.buckets[bucket_name]
            if not bucket_indices:
                continue
            n_samples = int(self._len * ratio)
            perm = torch.randperm(len(bucket_indices), generator=g)
            repeated = (perm.repeat(math.ceil(n_samples / len(bucket_indices))))[:n_samples]
            indices.extend([bucket_indices[i] for i in repeated.tolist()])

        # Shuffle all indices together
        perm = torch.randperm(len(indices), generator=g)
        return iter([indices[i] for i in perm.tolist()])


class TokenBudgetCallback(TrainerCallback):
    """Stop training after consuming token_budget real tokens."""

    def __init__(self, token_budget: int):
        self.token_budget = token_budget
        self._trainer = None

    def bind_trainer(self, trainer):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self._trainer and self._trainer.tokens_seen >= self.token_budget:
            control.should_training_stop = True


class AutoresearchCallback(TrainerCallback):
    """Live W&B logging + smoke abort during autoresearch runs."""

    def __init__(self, smoke_step=100):
        self.smoke_step = smoke_step
        self.smoke_done = False

    def _smoke_abort(self, reason, step, control):
        from research.tracker import alert
        import wandb
        wandb.run.summary["smoke_abort"] = True
        wandb.run.summary["abort_reason"] = reason
        wandb.run.summary["abort_step"] = step
        alert("SMOKE ABORT", reason)
        control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        from research.tracker import log
        if logs and "loss" in logs:
            loss = logs["loss"]
            log({"train_loss": loss}, step=state.global_step)

            if not self.smoke_done and state.global_step >= self.smoke_step:
                if math.isnan(loss) or math.isinf(loss):
                    self._smoke_abort(f"NaN/inf loss at step {state.global_step}",
                                      state.global_step, control)
                elif loss > 10.0:
                    self._smoke_abort(f"Loss {loss:.2f} > 10 at step {state.global_step}",
                                      state.global_step, control)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        from research.tracker import log
        from research.score import check_smoke
        if metrics:
            log(metrics, step=state.global_step)

        if not self.smoke_done and metrics:
            self.smoke_done = True
            passed, reason = check_smoke(metrics)
            if not passed:
                self._smoke_abort(reason, state.global_step, control)


class NerTrainer(Trainer):
    """Trainer with MixtureSampler support + token accounting."""

    def __init__(self, *args, mixture_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_sampler = mixture_sampler
        self.tokens_seen = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.tokens_seen += inputs["attention_mask"].sum().item()
        return super().training_step(model, inputs, num_items_in_batch)

    def get_train_dataloader(self):
        if self.mixture_sampler is not None:
            epoch = int(self.state.epoch) if self.state and self.state.epoch is not None else 0
            self.mixture_sampler.set_epoch(epoch)
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self.mixture_sampler,
                collate_fn=ner_collate_fn,
                num_workers=2,
                pin_memory=True,
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=ner_collate_fn,
            num_workers=2,
            pin_memory=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Train NER baseline")
    parser.add_argument("--train-path", default="data/processed/train.jsonl")
    parser.add_argument("--val-path", default="data/processed/val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/ner-baseline")
    parser.add_argument("--hub-model-id", default=None, help="HF Hub model ID for push_to_hub")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs for dry runs")
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--gold-ratio", type=float, default=GOLD_RATIO)
    parser.add_argument("--silver-en-ratio", type=float, default=SILVER_EN_RATIO)
    parser.add_argument("--silver-ml-ratio", type=float, default=SILVER_ML_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    type_mapping = load_type_mapping()

    train_dataset = NerDataset(args.train_path, tokenizer, type_mapping=type_mapping, is_train=True)
    val_dataset = NerDataset(args.val_path, tokenizer, type_mapping=type_mapping, is_train=False)

    model = NerModel(MODEL_NAME, NUM_LABELS, dropout=args.dropout)

    sampler = MixtureSampler(
        train_dataset,
        gold_ratio=args.gold_ratio,
        silver_en_ratio=args.silver_en_ratio,
        silver_ml_ratio=args.silver_ml_ratio,
        seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    callbacks = []
    is_autoresearch = os.environ.get("AUTORESEARCH_MODE")
    if is_autoresearch:
        from research.train import EXPERIMENT
        token_budget = EXPERIMENT.get("token_budget", 3_200_000)
        budget_cb = TokenBudgetCallback(token_budget)
        autoresearch_cb = AutoresearchCallback(smoke_step=EXPERIMENT.get("eval_steps", 100))
        callbacks = [budget_cb, autoresearch_cb]

        run_id = os.environ.get("AUTORESEARCH_RUN_ID", "")
        from research.tracker import init as tracker_init
        tracker_init(name=run_id, run_id=run_id, config=EXPERIMENT)

    trainer = NerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        mixture_sampler=sampler,
        callbacks=callbacks,
    )

    if is_autoresearch and callbacks:
        callbacks[0].bind_trainer(trainer)

    trainer.train()
    trainer.save_model()

    if is_autoresearch:
        from model.eval import run_full_eval
        device = next(model.parameters()).device
        eval_dataloader = trainer.get_eval_dataloader()
        metrics = run_full_eval(model, eval_dataloader, ID2LABEL, device)
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        metrics["num_steps"] = trainer.state.global_step
        metrics["tokens_seen"] = trainer.tokens_seen
        print("AUTORESEARCH_METRICS:" + json.dumps(metrics))

        from research.tracker import finish as tracker_finish
        tracker_finish()

    logger.info("Training complete. Best metric: %s", trainer.state.best_metric)


if __name__ == "__main__":
    main()
