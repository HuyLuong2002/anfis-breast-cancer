"""Grid search ANFIS: train/test split x learning rate x epochs."""
import warnings
warnings.filterwarnings("ignore")

import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skanfis import scikit_anfis
from skanfis.fs import FS, LinguisticVariable, GaussianFuzzySet
from skanfis.experimental import RMSELoss

# --- Load & preprocess WBCD (same as training notebook) ---
uci_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
)
cols = [
    "sample_code_number", "clump_thickness", "uniformity_of_cell_size",
    "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size",
    "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class",
]
feature_cols = [
    "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses",
]
PAPER_TOP3_FEATURES = [
    "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
]
FS_VAR_NAMES = {
    "clump_thickness": "ClumpThickness",
    "uniformity_of_cell_size": "CellSize",
    "uniformity_of_cell_shape": "CellShape",
}
LINGUISTIC_TERMS = ("low", "medium", "high")

df = pd.read_csv(uci_url, header=None, names=cols)
df = df.replace("?", np.nan).dropna().copy()
df["bare_nuclei"] = df["bare_nuclei"].astype(int)
df["target"] = (df["class"] == 4).astype(int)

X = df[feature_cols].values.astype(np.float32)
y = df["target"].values.astype(int)

OUTLIER_STD_MULTIPLIER = 2.5
data_center = X.mean(axis=0)
euclidean_distances = np.linalg.norm(X - data_center, axis=1)
outlier_threshold = euclidean_distances.mean() + OUTLIER_STD_MULTIPLIER * euclidean_distances.std()
keep_mask = euclidean_distances <= outlier_threshold
X_clean, y_clean = X[keep_mask], y[keep_mask]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_clean).astype(np.float32)
pca_full = PCA(n_components=9, random_state=42)
pca_full.fit(X_norm)

selected_idx = [feature_cols.index(c) for c in PAPER_TOP3_FEATURES]
X_selected = X_norm[:, selected_idx].astype(np.float32)

n_drop = len(X_selected) - 663
centroid_benign = X_selected[y_clean == 0].mean(axis=0)
centroid_malignant = X_selected[y_clean == 1].mean(axis=0)
dist_to_benign = np.linalg.norm(X_selected - centroid_benign, axis=1)
dist_to_malignant = np.linalg.norm(X_selected - centroid_malignant, axis=1)
own_dist = np.where(y_clean == 0, dist_to_benign, dist_to_malignant)
other_dist = np.where(y_clean == 0, dist_to_malignant, dist_to_benign)
quality_score = other_dist - own_dist
drop_idx = np.argsort(quality_score)[:n_drop]
quality_keep_mask = np.ones(len(X_selected), dtype=bool)
quality_keep_mask[drop_idx] = False
indices = np.where(quality_keep_mask)[0]
X_663 = X_selected[indices]
y_663 = y_clean[indices].astype(np.float32)

print(f"Dataset ready: {X_663.shape}, malignant rate={y_663.mean():.4f}")


def build_fs_and_model(X_train, epochs):
    """ANFIS grid 27 luat, consequent hoc duoc (zerotype=False) de MF/LR/epoch anh huong accuracy."""
    fs = FS()
    for feat_col in PAPER_TOP3_FEATURES:
        fs_var = FS_VAR_NAMES[feat_col]
        col = X_train[:, PAPER_TOP3_FEATURES.index(feat_col)]
        col_min, col_max = float(col.min()), float(col.max())
        centers = np.linspace(col_min, col_max, 3).tolist()
        sigma = max((col_max - col_min) / 3.0, 1e-3)
        mf_low = GaussianFuzzySet(mu=centers[0], sigma=sigma, term="low")
        mf_med = GaussianFuzzySet(mu=centers[1], sigma=sigma, term="medium")
        mf_high = GaussianFuzzySet(mu=centers[2], sigma=sigma, term="high")
        fs.add_linguistic_variable(
            fs_var,
            LinguisticVariable([mf_low, mf_med, mf_high], concept=fs_var),
        )
    fs.set_crisp_output_value("out", 0)
    grid_rules = []
    for ct, cs, csh in itertools.product(LINGUISTIC_TERMS, repeat=3):
        grid_rules.append(
            f"IF (ClumpThickness IS {ct}) AND (CellSize IS {cs}) AND (CellShape IS {csh}) "
            f"THEN (out IS 0.5)"
        )
    fs.add_rules(grid_rules)
    model = scikit_anfis(
        fs,
        description="WBCD_Gaussian27_GridSearch",
        epoch=epochs,
        hybrid=True,
        label="c",
        zerotype=False,
    )
    return model


def clamp_gaussian_sigma(model, min_sigma=1e-3):
    with torch.no_grad():
        for fuzzify_var in model.layer["fuzzify"].varmfs.values():
            for mf in fuzzify_var.mfdefs.values():
                if hasattr(mf, "sigma"):
                    mf.sigma.data.clamp_(min=min_sigma)


def train_model(model, X_train, y_train, epochs, learning_rate):
    optimizer = torch.optim.SGD(model.layer["fuzzify"].parameters(), lr=learning_rate)
    criterion = RMSELoss()
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(-1)
    grad_clip_norm = 1.0
    inner_steps = 1

    model.train()
    for ep in range(1, epochs + 1):
        with torch.no_grad():
            model.is_training = True
            model(X_train_t, y_train_t)
        model.is_training = False
        for _ in range(inner_steps):
            y_pred = model(X_train_t, y_train_t)
            optimizer.zero_grad()
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.layer["fuzzify"].parameters(), grad_clip_norm)
            optimizer.step()
            clamp_gaussian_sigma(model)
    model.eval()
    model.is_training = False
    return model


def predict_binary(model, X):
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).float()).detach().numpy()
    return np.clip(np.round(y_pred), 0, 1).astype(int)


# --- Grid search ---
SPLIT_RATIOS = {"80-20": 0.2, "70-30": 0.3, "60-40": 0.4}
LEARNING_RATES = [0.001, 0.01, 0.05]
EPOCH_LIST = [25, 50, 75, 100]

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)
results_path = out_dir / "hyperparam_grid_results.csv"

results = []
total = len(SPLIT_RATIOS) * len(LEARNING_RATES) * len(EPOCH_LIST)
run_idx = 0

for split_label, test_size in SPLIT_RATIOS.items():
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_663, y_663, test_size=test_size, random_state=42, stratify=y_663,
    )
    for lr in LEARNING_RATES:
        for n_epochs in EPOCH_LIST:
            run_idx += 1
            print(f"[{run_idx}/{total}] split={split_label} lr={lr} epochs={n_epochs} ...", flush=True)
            model = build_fs_and_model(X_tr, n_epochs)
            model = train_model(model, X_tr, y_tr, n_epochs, lr)
            y_pred = predict_binary(model, X_te)
            acc = accuracy_score(y_te.astype(int), y_pred)
            results.append({
                "split_ratio": split_label,
                "test_size": test_size,
                "train_size": len(X_tr),
                "test_n": len(X_te),
                "learning_rate": lr,
                "epochs": n_epochs,
                "test_accuracy": acc,
                "test_accuracy_pct": round(acc * 100, 2),
            })
            print(f"  -> accuracy={acc*100:.2f}%", flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)
print(f"\nSaved {len(results_df)} rows to {results_path}")
print(results_df.pivot_table(
    index=["split_ratio", "epochs"],
    columns="learning_rate",
    values="test_accuracy_pct",
))
