"""
Demo Streamlit: nhap 3 dac trung WBCD -> du doan ANFIS + giai thich luat mo.
Chay: streamlit run streamlit_app.py
"""
from __future__ import annotations

import itertools
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

warnings.filterwarnings("ignore")

from skanfis import scikit_anfis
from skanfis.fs import FS, GaussianFuzzySet, LinguisticVariable

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "best_model"

FEATURE_COLS_9 = [
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
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
FEATURE_VI = {
    "clump_thickness": "Độ dày cụm tế bào",
    "uniformity_of_cell_size": "Độ đồng đều kích thước tế bào",
    "uniformity_of_cell_shape": "Độ đồng đều hình dạng tế bào",
}
LINGUISTIC_TERMS = ("low", "medium", "high")


@st.cache_resource
def load_artifacts():
    with open(BEST / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(BEST / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(BEST / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    with open(BEST / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    rules_df = pd.read_csv(BEST / "fuzzy_rules.csv")
    mf_df = pd.read_csv(BEST / "membership_functions.csv")

    selected = meta["selected_features"]
    split_cfg = meta["splits"]["70-30"]
    mf_init = split_cfg["mf_init_log"]

    fs = FS()
    for feat_col, mf_info in zip(selected, mf_init):
        fs_var = FS_VAR_NAMES[feat_col]
        centers = mf_info["mu"]
        sigma = float(mf_info["sigma"])
        mfs = [
            GaussianFuzzySet(mu=float(centers[i]), sigma=sigma, term=term)
            for i, term in enumerate(LINGUISTIC_TERMS)
        ]
        fs.add_linguistic_variable(
            fs_var,
            LinguisticVariable(mfs, concept=FEATURE_DISPLAY[feat_col]),
        )
    fs.set_crisp_output_value("out", 0)
    grid_rules = [
        f"IF (ClumpThickness IS {ct}) AND (CellSize IS {cs}) AND (CellShape IS {csh}) THEN (out IS 0.5)"
        for ct, cs, csh in itertools.product(LINGUISTIC_TERMS, repeat=3)
    ]
    fs.add_rules(grid_rules)

    model = scikit_anfis(
        fs,
        description="WBCD_Streamlit_ANFIS_Demo",
        epoch=int(split_cfg["epochs"]),
        hybrid=True,
        label="c",
        zerotype=False,
    )
    model.load(str(BEST / "model.pt"))
    model.eval()
    model.is_training = False

    return {
        "meta": meta,
        "metrics": metrics,
        "info": info,
        "scaler": scaler,
        "model": model,
        "selected": selected,
        "rules_df": rules_df,
        "mf_df": mf_df,
        "split_cfg": split_cfg,
    }


def scale_three_features(scaler, selected: list[str], values: list[float]) -> np.ndarray:
    x9 = np.asarray(scaler.mean_, dtype=np.float64).copy()
    for feat, val in zip(selected, values):
        x9[FEATURE_COLS_9.index(feat)] = float(val)
    x_norm = scaler.transform(x9.reshape(1, -1)).astype(np.float32)
    idx = [FEATURE_COLS_9.index(c) for c in selected]
    return x_norm[:, idx]


def predict_one(model, x3: np.ndarray):
    x_t = torch.from_numpy(np.asarray(x3, dtype=np.float32)).float()
    model.eval()
    model.is_training = False
    with torch.no_grad():
        score = float(model(x_t).numpy().reshape(-1)[0])
        norm_w = model.weights.numpy().reshape(-1)
        rule_out = model.rule_tsk.numpy().reshape(-1)
    pred = int(np.clip(np.round(score), 0, 1))
    return score, pred, norm_w, rule_out


def gaussian_membership(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def dominant_membership(mf_df: pd.DataFrame, selected: list[str], x3: np.ndarray, raw_vals: list[int]) -> pd.DataFrame:
    """Mot dong / dac trung: gia tri nhap + nhan ngon ngu manh nhat."""
    rows = []
    for feat, x, raw in zip(selected, x3.reshape(-1), raw_vals):
        name = FEATURE_DISPLAY[feat]
        sub = mf_df[mf_df["Input"] == name]
        best_term, best_mu = "", -1.0
        for _, r in sub.iterrows():
            mu = gaussian_membership(float(x), float(r["mu (center)"]), float(r["sigma"]))
            if mu > best_mu:
                best_mu = mu
                best_term = str(r["Linguistic term"])
        rows.append(
            {
                "Đặc trưng": FEATURE_VI[feat],
                "Giá trị nhập (1–10)": int(raw),
                "Mức ngôn ngữ gần nhất": best_term,
            }
        )
    return pd.DataFrame(rows)


def top_rules_table(
    rules_df: pd.DataFrame,
    norm_w: np.ndarray,
    rule_out: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Sap xep theo |dong gop| — dung anh huong len diem cuoi, khong chi do kich hoat."""
    n = min(len(rules_df), len(norm_w), len(rule_out))
    contrib = norm_w[:n] * rule_out[:n]
    order = np.argsort(-np.abs(contrib))[:top_k]
    rows = []
    for i in order:
        i = int(i)
        row = rules_df.iloc[i]
        c = float(contrib[i])
        rows.append(
            {
                "Luật": int(row["Rule"]),
                "Điều kiện (IF)": row["IF"],
                "Mức kích hoạt": f"{float(norm_w[i]) * 100:.1f}%",
                "Đóng góp vào điểm": round(c, 2),
                "Kéo về": "Ác tính" if c > 0 else "Lành tính",
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="ANFIS WBCD Demo", layout="centered")
    st.title("Chẩn đoán ung thư vú bằng ANFIS")
    st.caption("Wisconsin Breast Cancer Dataset — 3 đặc trưng chính theo paper")

    if not (BEST / "model.pt").exists():
        st.error(f"Không tìm thấy mô hình trong `{BEST}`.")
        st.stop()

    art = load_artifacts()
    model = art["model"]
    scaler = art["scaler"]
    selected = art["selected"]
    test_acc = art["metrics"]["metrics"]["TEST"]["accuracy"]

    st.markdown(f"**Độ chính xác trên tập test:** {test_acc * 100:.1f}%")

    st.subheader("1. Nhập 3 đặc trưng")
    st.caption("Thang điểm tế bào học WBCD: 1 (thấp) → 10 (cao)")
    v1 = st.slider(FEATURE_VI[selected[0]], 1, 10, 5)
    v2 = st.slider(FEATURE_VI[selected[1]], 1, 10, 5)
    v3 = st.slider(FEATURE_VI[selected[2]], 1, 10, 5)

    if not st.button("Dự đoán", type="primary", use_container_width=True):
        st.stop()

    raw_vals = [v1, v2, v3]
    x3 = scale_three_features(scaler, selected, raw_vals)
    score, pred, norm_w, rule_out = predict_one(model, x3)
    label_vi = "Lành tính" if pred == 0 else "Ác tính"
    label_en = "benign" if pred == 0 else "malignant"

    st.subheader("2. Kết quả")
    if pred == 0:
        st.success(f"**{label_vi}** ({label_en})")
    else:
        st.error(f"**{label_vi}** ({label_en})")
    st.write(
        f"Điểm mô hình: **{score:.2f}** (ngưỡng **0.5**: dưới = lành tính, từ 0.5 = ác tính)."
    )

    st.subheader("3. Đặc trưng ở mức ngôn ngữ nào?")
    st.dataframe(
        dominant_membership(art["mf_df"], selected, x3, raw_vals),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("4. Luật nào ảnh hưởng mạnh nhất tới điểm?")
    st.caption(
        "ANFIS **không** lấy kết luận của một luật duy nhất. "
        "Điểm cuối = **tổng đóng góp** của cả 27 luật. "
        "Bảng dưới xếp theo |đóng góp| lớn nhất (ảnh hưởng thực), không chỉ mức kích hoạt."
    )
    st.dataframe(
        top_rules_table(art["rules_df"], norm_w, rule_out, top_k=5),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
