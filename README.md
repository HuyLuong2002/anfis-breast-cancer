# Anfis-breast-cancer

## Môi trường triển khai

Project này được thiết kế để chạy trong môi trường Python local hoặc Google Colab, với notebook chính là [anfis_breast_cancer_model_training.ipynb](anfis_breast_cancer_model_training.ipynb). Khi chạy local, nên tạo môi trường ảo `.venv` để cô lập phụ thuộc và tránh xung đột với các package đã cài sẵn trong máy.

- Hệ điều hành hỗ trợ: Windows, Linux hoặc macOS.
- Phiên bản Python khuyến nghị: 3.8 trở lên.
- Nền tảng thực thi: Jupyter Notebook hoặc Google Colab.
- Thư viện chính: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `ucimlrepo` và `scikit-anfis`.
- Cách cài đặt local: clone mã nguồn `scikit-anfis`, cài dependencies từ `requirements.txt`, sau đó cài package ở chế độ editable bằng `pip install -e ./scikit-anfis`.
- Ghi chú triển khai: notebook đã cấu hình sẵn để chạy thử nghiệm trên máy cá nhân và giảm lỗi hiển thị của custom estimator trong `scikit-anfis`.