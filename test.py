import itertools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from skanfis import scikit_anfis
from skanfis.fs import FS, LinguisticVariable, GaussianFuzzySet
from skanfis.experimental import train_anfis

# Load WBCD (UCI Original)
url = (
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
    "clump_thickness",           # v1
    "uniformity_of_cell_size",   # v2
    "uniformity_of_cell_shape",  # v3
]
FS_VAR_NAMES = {
    "clump_thickness": "ClumpThickness",
    "uniformity_of_cell_size": "CellSize",
    "uniformity_of_cell_shape": "CellShape",
}
FEATURE_DISPLAY = {
    "clump_thickness": "Clump Thickness",
    "uniformity_of_cell_size": "Uniformity of Cell Size",
    "uniformity_of_cell_shape": "Uniformity of Cell Shape",
}
LINGUISTIC_TERMS = ("low", "medium", "high")

df = pd.read_csv(url, header=None, names=cols)
df = df.replace("?", np.nan).dropna().copy()
df["bare_nuclei"] = df["bare_nuclei"].astype(int)
X = df[feature_cols].values.astype(np.float32)
y = (df["class"] == 4).astype(int).values

# Chuẩn hóa 9 đặc trưng rồi PCA để báo cáo % quan trọng (Bảng 8 paper)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X).astype(np.float32)

pca_full = PCA(n_components=9, random_state=42)
pca_full.fit(X_norm)

pca_table = pd.DataFrame({
    "Attribute": [f"v{i}" for i in range(1, 10)],
    "Feature": feature_cols,
    "Pct_Importance": (100.0 * pca_full.explained_variance_ratio_).round(4),
})
print("PCA Results (v1..v9 <-> PC1..PC9):")
print(pca_table.to_string(index=False))

selected_idx = [feature_cols.index(c) for c in PAPER_TOP3_FEATURES]
X_selected = X_norm[:, selected_idx].astype(np.float32)

top3_pct = pca_table.set_index("Feature").loc[PAPER_TOP3_FEATURES, "Pct_Importance"]
print("\n3 features for ANFIS:", PAPER_TOP3_FEATURES)
print("Cumulative PCA importance (%):", round(float(top3_pct.sum()), 4))

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)


def expert_diagnosis(clump_term, size_term, shape_term):
    """Chuyên gia gán nhãn: đặc trưng thấp -> lành tính, cao -> ác tính."""
    score = {"low": 0, "medium": 1, "high": 2}
    v = [score[clump_term], score[size_term], score[shape_term]]
    high_count = sum(x == 2 for x in v)
    if high_count >= 2:
        return "malignant"
    if sum(v) <= 2:
        return "benign"
    if v[0] == 2:
        return "malignant"
    if sum(v) >= 4:
        return "malignant"
    return "benign"


# FS() + Gaussian MF (low/medium/high) + 27 luật chuyên gia
fs = FS()
for feat_col in PAPER_TOP3_FEATURES:
    fs_var = FS_VAR_NAMES[feat_col]
    col = X_train[:, PAPER_TOP3_FEATURES.index(feat_col)]
    col_min, col_max = float(col.min()), float(col.max())
    centers = np.linspace(col_min, col_max, 3).tolist()
    sigma = max((col_max - col_min) / 3.0, 1e-3)
    fs.add_linguistic_variable(
        fs_var,
        LinguisticVariable(
            [
                GaussianFuzzySet(mu=centers[0], sigma=sigma, term="low"),
                GaussianFuzzySet(mu=centers[1], sigma=sigma, term="medium"),
                GaussianFuzzySet(mu=centers[2], sigma=sigma, term="high"),
            ],
            concept=FEATURE_DISPLAY[feat_col],
        ),
    )

fs.set_crisp_output_value("benign", 0)
fs.set_crisp_output_value("malignant", 1)

EXPERT_RULES = []
for ct, cs, csh in itertools.product(LINGUISTIC_TERMS, repeat=3):
    diagnosis = expert_diagnosis(ct, cs, csh)
    EXPERT_RULES.append(
        f"IF (ClumpThickness IS {ct}) AND (CellSize IS {cs}) AND (CellShape IS {csh}) "
        f"THEN (Diagnosis IS {diagnosis})"
    )
fs.add_rules(EXPERT_RULES)

print(f"\nFS: {len(EXPERT_RULES)} rules (grid 3^3)")

model = scikit_anfis(
    fs,
    description="WBCD_PCA_top3_ExpertRules",
    epoch=300,
    hybrid=True,
    label="c",
    zerotype=False,
)

X_t = torch.from_numpy(X_train).float()
y_t = torch.from_numpy(y_train).float()
loader = DataLoader(
    TensorDataset(X_t, torch.unsqueeze(y_t, -1)),
    batch_size=1024, shuffle=True,
)
train_anfis(model, loader, model.epoch, show_plots=True)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

print(f"\n=== {len(EXPERT_RULES)} expert fuzzy rules ===")
model.print_rules()
