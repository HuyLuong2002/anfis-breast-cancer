"""Generate per-split figures: CM (train+test) and loss curve (separate PNGs)."""
import warnings
warnings.filterwarnings("ignore")

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skanfis import scikit_anfis
from skanfis.fs import FS, LinguisticVariable, GaussianFuzzySet
from skanfis.experimental import RMSELoss

# Reuse preprocessing from run_hyperparam_grid.py
exec(open("run_hyperparam_grid.py", encoding="utf-8").read().split("# --- Grid search ---")[0])

SPLIT_RATIOS = {"80-20": 0.2, "70-30": 0.3, "60-40": 0.4}
SPLIT_TITLES = {
    "80-20": "80 - 20% of train-test data",
    "70-30": "70 - 30% of train-test data",
    "60-40": "60 - 40% of train-test data",
}
CM_LABELS = [
    ("True Negative", (0, 0)),
    ("False Positive", (0, 1)),
    ("False Negative", (1, 0)),
    ("True Positive", (1, 1)),
]

LEARNING_RATES = [0.001, 0.01, 0.05]
EPOCH_LIST = [25, 50, 75, 100]
LR_COLORS = {0.001: "#f4a261", 0.01: "#8ecae6", 0.05: "#b39ddb"}
LR_LABELS = {0.001: "0.001", 0.01: "0.01", 0.05: "0.05"}

grid_df = pd.read_csv(Path("models") / "hyperparam_grid_results.csv")
OUT_DIR = Path("models")


def best_config_for_split(split_label):
    """Chon lr + epoch co test accuracy cao nhat trong grid search."""
    sub = grid_df[grid_df["split_ratio"] == split_label]
    best = sub.loc[sub["test_accuracy"].idxmax()]
    return float(best["learning_rate"]), int(best["epochs"]), float(best["test_accuracy_pct"])


def train_with_history(model, X_train, y_train, epochs, learning_rate):
    optimizer = torch.optim.SGD(model.layer["fuzzify"].parameters(), lr=learning_rate)
    criterion = RMSELoss()
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    history = []
    model.train()
    for ep in range(1, epochs + 1):
        with torch.no_grad():
            model.is_training = True
            model(X_train_t, y_train_t)
        model.is_training = False
        y_pred = model(X_train_t, y_train_t)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.layer["fuzzify"].parameters(), 1.0)
        optimizer.step()
        clamp_gaussian_sigma(model)
        mse = torch.nn.functional.mse_loss(y_pred, y_train_t).item()
        history.append({"epoch": ep, "error": float(np.sqrt(mse))})
    model.eval()
    model.is_training = False
    return model, pd.DataFrame(history)


def plot_cm(ax, y_true, y_pred, title):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    sns.heatmap(
        cm, annot=False, cmap="YlOrRd", cbar=True, vmin=0,
        vmax=max(int(cm.max()), 1),
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        linewidths=2, linecolor="white", ax=ax,
    )
    for label, (r, c) in CM_LABELS:
        count = int(cm[r, c])
        pct = 100.0 * count / total if total else 0.0
        ax.text(c + 0.5, r + 0.5, f"{label}\n{count}\n{pct:.2f}%",
                ha="center", va="center", fontsize=10)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.text(
        0.5, -0.22,
        f"Accuracy = {acc:.3f}    Precision = {prec:.3f}    Recall = {rec:.3f}    F1 Score = {f1:.3f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
    )


def plot_grouped_accuracy(split_label, ax=None):
    """Bieu do cot accuracy (%) theo epoch / learning rate — tu grid search."""
    sub = grid_df[grid_df["split_ratio"] == split_label].copy()
    bar_width = 0.22
    x = np.arange(len(EPOCH_LIST))
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    else:
        fig = ax.figure
    for i, lr in enumerate(LEARNING_RATES):
        vals = [
            float(sub[(sub["epochs"] == ep) & (sub["learning_rate"] == lr)]["test_accuracy_pct"].iloc[0])
            for ep in EPOCH_LIST
        ]
        offset = (i - (len(LEARNING_RATES) - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width,
            color=LR_COLORS[lr], label=LR_LABELS[lr], edgecolor="none", alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.2f}", ha="center", va="bottom", rotation=90, fontsize=9,
            )
    ax.set_title(SPLIT_TITLES[split_label], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch at different learning rate", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in EPOCH_LIST])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


for split_label, test_size in SPLIT_RATIOS.items():
    lr, epochs, best_acc_pct = best_config_for_split(split_label)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_663, y_663, test_size=test_size, random_state=42, stratify=y_663,
    )
    model = build_fs_and_model(X_tr, epochs)
    model, hist = train_with_history(model, X_tr, y_tr, epochs, lr)
    y_tr_p = predict_binary(model, X_tr)
    y_te_p = predict_binary(model, X_te)

    slug = split_label.replace("-", "_")

    # Hinh 1: confusion matrix train + test
    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_cm(axes[0], y_tr, y_tr_p, "Confusion matrix for WDBC - train data")
    plot_cm(axes[1], y_te, y_te_p, "Confusion matrix for WDBC - test data")
    fig_cm.suptitle(
        f"{SPLIT_TITLES[split_label]}  |  best: lr={lr}, epochs={epochs}, test acc={best_acc_pct:.2f}%",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_cm = OUT_DIR / f"split_dashboard_{slug}.png"
    fig_cm.savefig(out_cm, dpi=200, bbox_inches="tight")
    plt.close(fig_cm)
    print(f"Saved: {out_cm}")

    # Hinh 2: epoch vs loss — chi config tot nhat (test accuracy cao nhat)
    best_rmse_idx = hist["error"].idxmin()
    best_rmse_ep = int(hist.loc[best_rmse_idx, "epoch"])
    best_rmse_val = float(hist.loc[best_rmse_idx, "error"])

    fig_loss, ax_e = plt.subplots(figsize=(9, 5.5))
    ax_e.plot(hist["epoch"], hist["error"], color="#1f77b4", linewidth=2, label="Train RMSE")
    ax_e.scatter([best_rmse_ep], [best_rmse_val], color="#2ca02c", s=80, zorder=5,
                 label=f"Best epoch={best_rmse_ep}, RMSE={best_rmse_val:.4f}")
    ax_e.set_title(
        f"Epoch vs Error (RMSE) — best config\n"
        f"{SPLIT_TITLES[split_label]} | lr={lr}, epochs={epochs}, test acc={best_acc_pct:.2f}%",
        fontweight="bold",
    )
    ax_e.set_xlabel("Epoch")
    ax_e.set_ylabel("Error (RMSE)")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="best")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    plt.tight_layout()
    out_loss = OUT_DIR / f"split_loss_{slug}.png"
    fig_loss.savefig(out_loss, dpi=200, bbox_inches="tight")
    plt.close(fig_loss)
    print(f"Saved: {out_loss}")

    # Hinh 3: bieu do cot accuracy theo LR/epoch (tu grid search)
    fig_bar, _ = plot_grouped_accuracy(split_label)
    plt.tight_layout()
    out_bar = OUT_DIR / f"hyperparam_accuracy_{slug}.png"
    fig_bar.savefig(out_bar, dpi=200, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"Saved: {out_bar}")
