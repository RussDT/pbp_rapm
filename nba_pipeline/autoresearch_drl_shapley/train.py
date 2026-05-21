#!/usr/bin/env python3
"""
Editable research surface for the DRL/Shapley value model.

Only this file should be modified during autoresearch loops.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error

import prepare
import train_drl_shapley as base


# Research surface
SEED = 7
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
LINEUP_DIM = 20
NUM_HEADS = 8
BATCH_SIZE = 512
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 7e-4
MAX_EPOCHS = 24
PATIENCE = 3
DROPOUT = 0.0
GRAD_CLIP = 1.0
TARGET_SYNC_EVERY = 1
MAX_SECONDS = float(os.environ.get("DRL_AR_MAX_SECONDS", "180"))


class ResearchValueModel(nn.Module):
    def __init__(self, num_players: int) -> None:
        super().__init__()
        self.player_embedding = nn.Embedding(num_players + 1, EMBEDDING_DIM, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 8)
        self.pos_embedding = nn.Embedding(len(base.POSITION_GROUPS), 8)
        self.token_proj = nn.Linear(EMBEDDING_DIM + 8 + 8, EMBEDDING_DIM)
        self.token_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.lineup_attn = nn.MultiheadAttention(EMBEDDING_DIM, num_heads=NUM_HEADS, batch_first=True, dropout=DROPOUT)
        self.lineup_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.lineup_head = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, LINEUP_DIM),
        )
        self.numeric_head = nn.Sequential(
            nn.Linear(len(base.NUMERIC_FEATURES), HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.SiLU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + LINEUP_DIM, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, len(base.SUPPORT)),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        player_ids: torch.Tensor,
        side_ids: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat(
            [
                self.player_embedding(player_ids),
                self.side_embedding(side_ids),
                self.pos_embedding(pos_ids),
            ],
            dim=-1,
        )
        tokens = self.token_norm(self.token_proj(tokens))
        attn_out, _ = self.lineup_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.lineup_norm(tokens + attn_out)
        lineup_summary = self.lineup_head(tokens.mean(dim=1))
        numeric_summary = self.numeric_head(numeric)
        encoded = torch.cat([numeric_summary, lineup_summary], dim=-1)
        return self.output_head(encoded)


def clone_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parameter_count(model: nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1_000_000.0


def safe_evaluate(model: nn.Module, prepared: prepare.PreparedExperiment, split: str, device: torch.device):
    try:
        return prepare.evaluate_model(model, prepared, split=split, device=device), ""
    except Exception as exc:  # pragma: no cover - defensive loop guard
        error_message = str(exc)
        if "y_proba contains values greater than 1" in error_message:
            try:
                return stable_evaluate_with_clipped_probs(model, prepared, split=split, device=device), "fallback_clipped_probs"
            except Exception as fallback_exc:  # pragma: no cover - defensive loop guard
                return None, f"{error_message}; fallback failed: {fallback_exc}"
        return None, error_message


def stable_evaluate_with_clipped_probs(
    model: nn.Module,
    prepared: prepare.PreparedExperiment,
    split: str,
    device: torch.device,
) -> dict:
    if split == "research":
        indices = prepared.research_valid_idx
        baselines = prepared.research_baselines
    elif split == "audit":
        indices = prepared.audit_valid_idx
        baselines = prepared.audit_baselines
    else:
        raise ValueError(f"Unknown split: {split}")

    probs, remaining_margin, entropy = base.predict_value_distribution(model, prepared.bundle, indices, device)
    probs = base.sanitize_probability_tensor(torch.as_tensor(probs, dtype=torch.float32)).cpu().numpy()

    pred_final_margin = prepared.bundle.pre_margin_home[indices] + remaining_margin
    observed_final_margin = prepared.bundle.terminal_final_margin_home[indices]
    pre_margin_tensor = torch.as_tensor(prepared.bundle.pre_margin_home[indices], dtype=torch.float32)
    win_probs = base.margin_to_win_prob(torch.as_tensor(probs, dtype=torch.float32), pre_margin_tensor).numpy()
    win_probs = np.clip(win_probs, 1e-6, 1 - 1e-6)
    observed_home_win = prepared.bundle.home_win[indices]

    entropy_corr = np.corrcoef(entropy, (observed_final_margin - pred_final_margin) ** 2)[0, 1]
    metrics = {
        "rmse_final_margin": float(np.sqrt(mean_squared_error(observed_final_margin, pred_final_margin))),
        "brier": float(brier_score_loss(observed_home_win, win_probs)),
        "logloss": float(log_loss(observed_home_win, win_probs, labels=[0, 1])),
        "entropy_variance_corr": float(entropy_corr if np.isfinite(entropy_corr) else 0.0),
    }
    metrics["score"] = prepare.compute_score(metrics, baselines)
    return metrics


def train_once() -> dict:
    t0 = time.time()
    prepare.ensure_results_tsv()
    prepared = prepare.ensure_prepared(rebuild_dataset=False)
    base.seed_everything(SEED)
    device = base.determine_device("auto")
    torch.set_float32_matmul_precision("high")

    num_players = int(prepared.bundle.player_ids.max())
    model = ResearchValueModel(num_players=num_players).to(device)
    target_model = ResearchValueModel(num_players=num_players).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)

    best_state = clone_state_dict(model)
    best_metrics, eval_error = safe_evaluate(model, prepared, split="research", device=device)
    if best_metrics is None:
        raise RuntimeError(f"Initial evaluation failed: {eval_error}")
    if eval_error:
        evaluation_error = eval_error
    epochs_without_improve = 0
    deadline = t0 + MAX_SECONDS
    train_seconds = 0.0
    epochs_ran = 0
    evaluation_error = eval_error

    for epoch in range(1, MAX_EPOCHS + 1):
        if time.time() >= deadline:
            break
        epoch_start = time.time()
        epochs_ran = epoch
        model.train()
        losses = []
        for batch in base.batch_indices(prepared.train_idx, BATCH_SIZE, shuffle=True, rng=rng):
            if time.time() >= deadline:
                break
            batch_data = base.gather_batch(prepared.bundle, batch, device)
            logits = model(
                batch_data["numeric"],
                batch_data["player_ids"],
                batch_data["side_ids"],
                batch_data["pos_ids"],
            )
            log_probs = F.log_softmax(logits, dim=1)
            with torch.no_grad():
                next_logits = target_model(
                    batch_data["next_numeric"],
                    batch_data["next_player_ids"],
                    batch_data["next_side_ids"],
                    batch_data["next_pos_ids"],
                )
                next_probs = F.softmax(next_logits, dim=1)
                target_probs = base.project_distribution(
                    next_probs=next_probs,
                    rewards=batch_data["rewards"],
                    gamma=batch_data["gamma"],
                    support=base.SUPPORT_TENSOR.to(device),
                    terminal=batch_data["terminal"],
                )
            loss = -(target_probs * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.item()))

        train_seconds += time.time() - epoch_start
        if epoch % TARGET_SYNC_EVERY == 0:
            target_model.load_state_dict(model.state_dict())

        research_metrics, eval_error = safe_evaluate(model, prepared, split="research", device=device)
        if research_metrics is None:
            evaluation_error = eval_error
            break
        if eval_error:
            evaluation_error = eval_error
        if research_metrics["score"] < best_metrics["score"]:
            best_metrics = research_metrics
            best_state = clone_state_dict(model)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

        if time.time() >= deadline:
            break

    model.load_state_dict(best_state)
    research_metrics, eval_error = safe_evaluate(model, prepared, split="research", device=device)
    if research_metrics is None:
        raise RuntimeError(f"Best-state research evaluation failed: {eval_error}")
    if eval_error:
        evaluation_error = eval_error
    audit_metrics, eval_error = safe_evaluate(model, prepared, split="audit", device=device)
    if audit_metrics is None:
        raise RuntimeError(f"Best-state audit evaluation failed: {eval_error}")
    if eval_error:
        evaluation_error = eval_error

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "latest_model.pt"
    torch.save(model.state_dict(), model_path)

    total_seconds = time.time() - t0
    summary = {
        "research_score": float(research_metrics["score"]),
        "audit_score": float(audit_metrics["score"]),
        "research_rmse": float(research_metrics["rmse_final_margin"]),
        "research_brier": float(research_metrics["brier"]),
        "research_logloss": float(research_metrics["logloss"]),
        "audit_rmse": float(audit_metrics["rmse_final_margin"]),
        "audit_brier": float(audit_metrics["brier"]),
        "audit_logloss": float(audit_metrics["logloss"]),
        "entropy_variance_corr": float(research_metrics["entropy_variance_corr"]),
        "training_seconds": float(train_seconds),
        "total_seconds": float(total_seconds),
        "epochs_ran": int(epochs_ran),
        "num_params_M": float(parameter_count(model)),
        "device": str(device),
        "max_seconds_budget": float(MAX_SECONDS),
        "target_score": float(prepare.TARGETS["research_score"]),
        "target_audit_score": float(prepare.TARGETS["audit_score"]),
        "model_path": str(model_path),
        "evaluation_error": evaluation_error,
    }
    prepare.save_summary(summary)
    return summary


def main() -> None:
    summary = train_once()
    print("---")
    for key in [
        "research_score",
        "audit_score",
        "research_rmse",
        "research_brier",
        "research_logloss",
        "audit_rmse",
        "audit_brier",
        "audit_logloss",
        "entropy_variance_corr",
        "training_seconds",
        "total_seconds",
        "epochs_ran",
        "num_params_M",
        "target_score",
        "target_audit_score",
    ]:
        print(f"{key}: {summary[key]:.6f}" if isinstance(summary[key], float) else f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
