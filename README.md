# ANFIS Breast Cancer (WBCD)

Dự án huấn luyện và đánh giá mô hình **ANFIS** (Adaptive Neuro-Fuzzy Inference System) trên bộ dữ liệu **Wisconsin Breast Cancer Dataset (WBCD)** — UCI Original. Pipeline bám theo paper *"Breast Cancer Diagnosis Based on Genetic-Fuzzy Logic and ANFIS Using WBCD"* (xem thư mục `paper/`).

Môi trường chạy: Python local (khuyến nghị `.venv`) hoặc Google Colab. Thực thi qua **Jupyter Notebook**.

---

## Luồng nghiệp vụ chính

```
Tiền xử lý WBCD → Grid search → Lưu CSV (models/) + biểu đồ (plots/) + nhật ký HTML (nhat-ky/)
                                              ↓
              Dự đoán toàn tập (predict) · Baseline trước train · (tuỳ chọn) vẽ lại biểu đồ
```

| Thứ tự | Notebook | Vai trò |
|--------|----------|---------|
| 1 | [anfis_wbcd_pretrain_baseline.ipynb](anfis_wbcd_pretrain_baseline.ipynb) | **Baseline trước huấn luyện** — kiểm tra mô hình `scikit-anfis` dự đoán ra sao ngay sau khởi tạo, **chưa gọi `fit()`** |
| 2 | [anfis_pca_scikit_anfis_training.ipynb](anfis_pca_scikit_anfis_training.ipynb) | **Huấn luyện chính** — grid search, lưu CSV/biểu đồ/nhật ký |
| 3 | [anfis_pca_predict_full_wbcd.ipynb](anfis_pca_predict_full_wbcd.ipynb) | **Inference** — load mô hình tốt nhất, dự đoán trên **toàn bộ** tập WBCD |
| 4 | [plot_split_dashboard.ipynb](plot_split_dashboard.ipynb) | **(Tuỳ chọn)** Vẽ lại biểu đồ từ CSV đã có, không cần chạy lại grid search |

---

## Chi tiết từng notebook

### 1. `anfis_wbcd_pretrain_baseline.ipynb` — Baseline trước training

**Mục đích:** Đo hiệu năng “thô” của ANFIS trước khi học tham số, làm mốc so sánh với mô hình đã train.

**Nội dung chính:**
- Tiền xử lý WBCD giống pipeline training (loại missing, outlier, chọn 3 feature theo paper: v1/v2/v3)
- Split theo paper: **200 train / 263 check / 200 test**
- Hệ mờ `FS()` với **27 luật chuyên gia** + hàm thành viên Gaussian
- Gọi `predict()` ngay sau khởi tạo — **không huấn luyện**
- (Tuỳ chọn) So sánh với checkpoint đã lưu trong `models/`

**Khi nào chạy:** Trước hoặc song song với training, để chứng minh mô hình cải thiện sau khi `fit()`.

---

### 2. `anfis_pca_scikit_anfis_training.ipynb` — Huấn luyện

**Mục đích:** Notebook trung tâm để huấn luyện và tìm hyperparameter tốt.

**Nội dung chính:**

| Bước | Mô tả |
|------|--------|
| 0 | Import & cấu hình (`models/`, `plots/`, `nhat-ky/`) |
| 1 | Tiền xử lý WBCD → 663 mẫu, 3 feature PCA |
| 2 | Hệ mờ FS: MF Gaussian + **27 luật** grid |
| 3 | Grid search: 3 split × 3 LR × 4 epoch = **36 cấu hình** |
| 4 | Bảng kết quả & pivot accuracy |
| 5 | Vẽ biểu đồ & **lưu 4 PNG/split** vào `plots/` |
| 6 | Tóm tắt cấu hình tốt nhất mỗi split |
| 7 | **Xuất nhật ký HTML** vào `nhat-ky/` |

**Đầu ra sau khi chạy hết notebook:**

| Loại | Vị trí | File |
|------|--------|------|
| Grid search | `models/` | `hyperparam_grid_results.csv` |
| Biểu đồ (×4 mỗi split) | `plots/` | `split_dashboard_*.png`, `split_cm_*.png`, `split_loss_*.png`, `hyperparam_accuracy_*.png` |
| Nhật ký | `nhat-ky/` | `{timestamp}_grid_search_pca_feature_select_ANFIS_grid_80_20_70_30_60_40.html` |

**Ghi chú:** Các lần train paper-style (split 200/263/200) từ phiên bản trước vẫn lưu checkpoint `{timestamp}_paperstyle_*` trong `models/`. Mô hình tốt nhất cho inference là stamp **`20260621_174830`** (notebook predict).

---

### 3. `anfis_pca_predict_full_wbcd.ipynb` — Dự đoán toàn tập (mô hình tốt nhất)

**Mục đích:** Load checkpoint đã huấn luyện và chạy inference trên **toàn bộ** WBCD (683 mẫu sau loại missing, không áp dụng lại bước giảm 663 mẫu của split paper).

**Pipeline inference (bám đúng lúc train):**
1. Chuẩn hóa 9 feature bằng `scaler.pkl` đã lưu
2. Chọn 3 feature cố định theo paper (v1, v2, v3)
3. Load ANFIS checkpoint + `predict`
4. Đánh giá metric và lưu kết quả

**Cấu hình mô hình mặc định** (cell cấu hình trong notebook):

| Tham số | Giá trị |
|---------|---------|
| `STAMP` | `20260621_174830` |
| `RUN_TAG` | `paperstyle_pca_feature_select` |

**Đầu ra:** `models/20260621_174830_full_wbcd_predictions.csv` — dự đoán từng mẫu (sample code, nhãn thật, nhãn dự đoán, xác suất).

---

### 4. `plot_split_dashboard.ipynb` — Vẽ lại biểu đồ (tuỳ chọn)

**Mục đích:** Notebook độc lập để **vẽ lại** biểu đồ từ `models/hyperparam_grid_results.csv` mà không cần chạy lại toàn bộ grid search trong notebook training.

**Khi nào dùng:**
- Đã có CSV kết quả grid search, chỉ muốn chỉnh style biểu đồ hoặc xuất lại PNG
- Không muốn chạy lại mục 5–7 của notebook training

**Lưu ý:** Notebook training (`anfis_pca_scikit_anfis_training.ipynb`) **đã tự lưu đủ biểu đồ** vào `plots/` sau mục 5. Thông thường **không cần** chạy notebook này nếu đã chạy hết notebook training.

---

## Cài đặt thư viện

### Yêu cầu hệ thống

- **OS:** Windows, Linux hoặc macOS
- **Python:** 3.8 trở lên (khuyến nghị 3.10+)
- **Nền tảng:** Jupyter Notebook / JupyterLab / Google Colab

### Thư viện chính

| Thư viện | Vai trò |
|----------|---------|
| `numpy`, `pandas`, `scipy` | Xử lý dữ liệu |
| `scikit-learn` | PCA, chuẩn hóa, metric, split |
| `torch` | Backend huấn luyện ANFIS |
| `matplotlib`, `seaborn` | Biểu đồ |
| `scikit-anfis` | ANFIS estimator (fork local trong `scikit-anfis/`) |
| `jupyter`, `nbconvert` | Chạy notebook, xuất HTML nhật ký |

### Cài đặt local (Windows / Linux / macOS)

```bash
# 1. Clone repo và vào thư mục dự án
cd anfis-breast-cancer

# 2. Tạo môi trường ảo
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Cài dependencies (bao gồm scikit-anfis ở chế độ editable)
pip install -r requirements.txt
```

`requirements.txt` đã khai báo `-e ./scikit-anfis` — package ANFIS được cài trực tiếp từ mã nguồn trong repo, không cần clone riêng.

### Chạy notebook

```bash
jupyter notebook
# hoặc
jupyter lab
```

Mở notebook theo thứ tự nghiệp vụ ở bảng trên. Lần đầu chạy training cần kết nối internet để tải WBCD từ UCI.

---

## Thư mục `nhat-ky/` — Nhật ký chạy thí nghiệm

**Mục đích:** Lưu **bản HTML** của notebook training sau mỗi lần chạy — dùng để **tra cứu, audit và báo cáo** mà không cần mở lại file `.ipynb`.

**Cách tạo (tự động):** Chạy **mục 7** cuối notebook `anfis_pca_scikit_anfis_training.ipynb`. Cell sẽ gọi `jupyter nbconvert` và lưu file vào `nhat-ky/`.

**Quy ước tên file (grid search hiện tại):**

```
{YYYYMMDD_HHMMSS}_grid_search_pca_feature_select_ANFIS_grid_80_20_70_30_60_40.html
```

**Quy ước tên file (các lần chạy paper-style cũ):**

```
{YYYYMMDD_HHMMSS}_paperstyle_pca_feature_select_ANFIS_-_split_200_263_200.html
```

Mỗi file HTML là “ảnh chụp” đầy đủ code, output và biểu đồ tại thời điểm chạy.

---

## Thư mục `models/` — Artifact mô hình & kết quả

**Mục đích:** Kho lưu trữ **tất cả sản phẩm** từ huấn luyện, grid search và inference. Mỗi lần train paper-style có **stamp thời gian** (`YYYYMMDD_HHMMSS`) để phân biệt các run.

### Nhóm file theo stamp (ví dụ `20260621_174830`)

| File | Mô tả |
|------|--------|
| `{stamp}_paperstyle_pca_feature_select_best_model.pkl` | **Checkpoint ANFIS** — mô hình đã huấn luyện (dùng cho inference) |
| `{stamp}_paperstyle_scaler.pkl` | `StandardScaler` đã fit trên 9 feature — bắt buộc khi predict |
| `{stamp}_paperstyle_meta.json` | Metadata đầy đủ: feature đã chọn, split, hyperparameter, 27 luật mờ, MF init |
| `{stamp}_paperstyle_metrics.json` / `.csv` | Metric trên tập CHECK và TEST (accuracy, precision, recall, F1, ROC-AUC) |
| `{stamp}_paperstyle_gaussian27_training_loss_log.csv` | Log RMSE theo epoch khi train |
| `{stamp}_paperstyle_membership_functions.csv` | Tham số hàm thành viên Gaussian sau train |
| `{stamp}_paperstyle_fuzzy_rules.csv` | 27 luật mờ (IF-THEN) |
| `{stamp}_paperstyle_pca_info.pkl` | Thông tin PCA (explained variance, v.v.) |
| `{stamp}_paperstyle_outlier_dropped_samples.csv` | Mẫu bị loại do outlier Euclidean |
| `{stamp}_paperstyle_quality_dropped_samples.csv` | Mẫu bị loại do quality-based drop (663 mẫu) |
| `{stamp}_full_wbcd_predictions.csv` | Kết quả dự đoán toàn tập WBCD (từ notebook predict) |

### File grid search (không theo stamp)

| File | Mô tả |
|------|--------|
| `hyperparam_grid_results.csv` | Kết quả 36 cấu hình grid search |

### Biểu đồ grid search (trong `plots/`)

| File | Mô tả |
|------|--------|
| `split_dashboard_*.png` | Dashboard tổng hợp (CM + loss + accuracy) |
| `split_cm_*.png` | Confusion matrix train + test |
| `split_loss_*.png` | Biểu đồ loss RMSE |
| `hyperparam_accuracy_*.png` | Accuracy theo LR và epoch |

> Các file PNG cũ có thể nằm trong `models/` từ lần chạy trước; phiên bản hiện tại lưu vào `plots/`.

### Mô hình tốt nhất hiện tại

Stamp **`20260621_174830`** — accuracy TEST ~93.5%, CHECK ~94.3% (xem `paperstyle_metrics.json`). Notebook predict trỏ mặc định vào stamp này.

---

## Cấu trúc thư mục dự án

```
anfis-breast-cancer/
├── anfis_wbcd_pretrain_baseline.ipynb      # Baseline trước training
├── anfis_pca_scikit_anfis_training.ipynb   # Huấn luyện + grid search
├── anfis_pca_predict_full_wbcd.ipynb       # Dự đoán toàn tập WBCD
├── plot_split_dashboard.ipynb              # Vẽ biểu đồ trình bày
├── requirements.txt                        # Dependencies Python
├── scikit-anfis/                           # Fork thư viện ANFIS (editable install)
├── models/                                 # Checkpoint, metric, biểu đồ, CSV kết quả
├── nhat-ky/                                # HTML nhật ký từng lần chạy training
├── plots/                                  # Biểu đồ xuất từ notebook training (mục 5)
└── paper/                                  # PDF paper tham chiếu
```

---

## Ghi chú triển khai

- Notebook đã cấu hình giảm cảnh báo hiển thị của custom estimator trong `scikit-anfis`.
- Dữ liệu WBCD tải trực tiếp từ UCI Archive khi chạy notebook (cần internet lần đầu).
- Thư mục `.venv/` nên được gitignore — mỗi máy tạo môi trường riêng qua `requirements.txt`.
