# Machine Failure Prediction

Ứng dụng dự đoán nguy cơ hỏng hóc máy móc công nghiệp sử dụng các thuật toán Machine Learning.

## 1. Giới thiệu

Dự án xây dựng hệ thống dự đoán tình trạng hỏng hóc của máy móc dựa trên các thông số vận hành.

Hệ thống sử dụng bộ dữ liệu AI4I 2020 Predictive Maintenance Dataset và triển khai nhiều thuật toán học máy nhằm so sánh hiệu quả dự đoán.

## 2. Công nghệ sử dụng

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

## 3. Các thuật toán

Dự án thử nghiệm và so sánh 5 thuật toán:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

Các chỉ số đánh giá gồm:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## 4. Chức năng ứng dụng

### Dashboard
Hiển thị tổng quan dữ liệu và tỷ lệ máy bình thường, máy hỏng.

### Dự đoán một máy
Nhập thông số vận hành của một máy để nhận kết quả dự đoán.

### Dự đoán hàng loạt
Tải lên file CSV để dự đoán nhiều máy cùng lúc và xuất kết quả.

### So sánh mô hình
Hiển thị bảng chỉ số, biểu đồ so sánh và Confusion Matrix.

## 5. Cấu trúc dự án

```text
MachineFailurePrediction/
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── best_model.pkl
│   └── metadata.json
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   └── model_comparison.csv
├── src/
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Cài đặt

Yêu cầu: Python 3.10 trở lên.

Clone repository:

```bash
git clone <GITHUB_REPOSITORY_URL>
```

Di chuyển vào thư mục dự án:

```bash
cd MachineFailurePrediction
```

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```powershell
.venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 7. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại:

```text
http://localhost:8501
```

## 8. Dataset

AI4I 2020 Predictive Maintenance Dataset.

Nguồn:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## 9. Lưu ý

Kết quả dự đoán phụ thuộc vào dữ liệu và mô hình đã huấn luyện. Xác suất đầu ra là ước lượng của mô hình, không phải xác nhận chắc chắn tình trạng thực tế của máy móc.

Dự án phục vụ mục đích học tập và nghiên cứu.
