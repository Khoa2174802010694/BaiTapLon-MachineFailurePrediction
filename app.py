
import os
import json
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. CẤU HÌNH
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
RESULT_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")
CM_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.csv")

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

NUMERIC_FEATURES = FEATURES[1:]

FEATURE_LABELS = {
    "Type": "Loại máy",
    "Air temperature [K]": "Nhiệt độ không khí (K)",
    "Process temperature [K]": "Nhiệt độ quy trình (K)",
    "Rotational speed [rpm]": "Tốc độ quay (rpm)",
    "Torque [Nm]": "Mô-men xoắn (Nm)",
    "Tool wear [min]": "Độ mài mòn dụng cụ (phút)"
}

MODEL_LABELS = {
    "Logistic Regression": "Hồi quy Logistic",
    "KNN": "K-Nearest Neighbors",
    "Decision Tree": "Cây quyết định",
    "Random Forest": "Rừng ngẫu nhiên",
    "SVM": "Support Vector Machine"
}

st.set_page_config(
    page_title="Dự đoán hỏng hóc máy móc",
    page_icon="⚙️",
    layout="wide"
)


# =====================================================
# 2. TẢI MÔ HÌNH VÀ DỮ LIỆU
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(RESULT_PATH)


@st.cache_data
def load_confusion_matrix():
    return pd.read_csv(CM_PATH, index_col=0)


try:
    model = load_model()
    df = load_data()
    comparison = load_comparison()
except FileNotFoundError as error:
    st.error(f"Không tìm thấy tệp cần thiết: {error}")
    st.stop()
except Exception as error:
    st.error(f"Không thể tải dữ liệu hoặc mô hình: {error}")
    st.stop()


# =====================================================
# 3. THANH ĐIỀU HƯỚNG
# =====================================================

st.sidebar.title("⚙️ Machine Failure")
st.sidebar.caption("Hệ thống dự đoán hỏng hóc máy móc")

page = st.sidebar.radio(
    "DANH MỤC",
    [
        "Tổng quan",
        "Dự đoán một máy",
        "Dự đoán hàng loạt",
        "So sánh mô hình"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Machine Failure Prediction**

    Ứng dụng sử dụng các thuật toán
    học máy để dự đoán nguy cơ hỏng hóc
    dựa trên thông số vận hành máy móc.
    """
)

st.sidebar.caption("Đồ án môn Học máy và ứng dụng")


# =====================================================
# 4. HÀM HỖ TRỢ
# =====================================================

def show_prediction_result(prediction, probability):
    st.subheader("Kết quả dự đoán")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ CÓ NGUY CƠ HỎNG HÓC")
            st.write(
                "Mô hình phân loại thiết bị vào nhóm "
                "có nguy cơ hỏng hóc."
            )
        else:
            st.success("✅ THIẾT BỊ BÌNH THƯỜNG")
            st.write(
                "Mô hình phân loại thiết bị vào nhóm "
                "bình thường."
            )

    with col2:
        st.metric(
            "Điểm xác suất dự đoán hỏng",
            f"{probability * 100:.2f}%"
        )
        st.progress(float(probability))

    st.caption(
        "Đây là xác suất do mô hình ước lượng, "
        "không phải xác nhận chắc chắn tình trạng "
        "thực tế của thiết bị."
    )


def validate_input_data(input_df):
    missing_columns = [
        column for column in FEATURES
        if column not in input_df.columns
    ]

    if missing_columns:
        return False, (
            "File thiếu các cột bắt buộc: "
            + ", ".join(missing_columns)
        )

    if input_df.empty:
        return False, "File CSV không có bản ghi nào."

    data = input_df[FEATURES].copy()

    if data.isnull().any().any():
        return False, (
            "Dữ liệu có ô trống. "
            "Vui lòng bổ sung trước khi dự đoán."
        )

    valid_types = ["L", "M", "H"]

    invalid_types = ~data["Type"].astype(str).isin(valid_types)

    if invalid_types.any():
        return False, (
            "Cột Type chỉ chấp nhận các giá trị L, M hoặc H."
        )

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[NUMERIC_FEATURES].isnull().any().any():
        return False, (
            "Một hoặc nhiều cột số chứa dữ liệu "
            "không hợp lệ. Vui lòng kiểm tra lại."
        )

    if not data[NUMERIC_FEATURES].apply(
        lambda column: column.map(
            lambda value: pd.notna(value)
            and abs(value) != float("inf")
        )
    ).all().all():
        return False, "Dữ liệu chứa giá trị vô hạn hoặc không hợp lệ."

    return True, data


# =====================================================
# 5. TRANG TỔNG QUAN
# =====================================================

if page == "Tổng quan":

    st.title("Tổng quan hệ thống")
    st.write(
        "Theo dõi dữ liệu và tình trạng hỏng hóc "
        "trong bộ dữ liệu AI4I 2020."
    )

    total = len(df)
    failures = int(df["Machine failure"].sum())
    normal = total - failures
    failure_rate = failures / total * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tổng số bản ghi", f"{total:,}")
    col2.metric("Máy bình thường", f"{normal:,}")
    col3.metric("Máy bị hỏng", f"{failures:,}")
    col4.metric("Tỷ lệ hỏng", f"{failure_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Tỷ lệ trạng thái máy")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            [normal, failures],
            labels=["Bình thường", "Hỏng hóc"],
            autopct="%1.2f%%",
            startangle=90
        )

        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Số lượng theo trạng thái")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            ["Bình thường", "Hỏng hóc"],
            [normal, failures]
        )

        ax.set_ylabel("Số lượng máy")
        ax.set_title("Phân phối trạng thái máy")

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Xem trước dữ liệu")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Nguồn dữ liệu: AI4I 2020 Predictive Maintenance Dataset."
    )


# =====================================================
# 6. DỰ ĐOÁN MỘT MÁY
# =====================================================

elif page == "Dự đoán một máy":

    st.title("Dự đoán tình trạng một máy")

    st.write(
        "Nhập các thông số vận hành để hệ thống "
        "đưa ra dự đoán."
    )

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox(
                "Loại máy",
                ["L", "M", "H"],
                help="L: Low, M: Medium, H: High"
            )

            air_temp = st.number_input(
                "Nhiệt độ không khí (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Nhiệt độ quy trình (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

        with col2:
            rotational_speed = st.number_input(
                "Tốc độ quay (rpm)",
                min_value=100.0,
                max_value=5000.0,
                value=1500.0,
                step=10.0
            )

            torque = st.number_input(
                "Mô-men xoắn (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1
            )

            tool_wear = st.number_input(
                "Độ mài mòn dụng cụ (phút)",
                min_value=0.0,
                max_value=300.0,
                value=100.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "🔍 Dự đoán tình trạng máy",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        try:
            prediction = int(model.predict(input_data)[0])
            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            st.divider()
            show_prediction_result(prediction, probability)

        except Exception as error:
            st.error(f"Không thể thực hiện dự đoán: {error}")


# =====================================================
# 7. DỰ ĐOÁN HÀNG LOẠT
# =====================================================

elif page == "Dự đoán hàng loạt":

    st.title("Dự đoán hàng loạt")

    st.write(
        "Tải lên tệp CSV chứa thông số vận hành "
        "của nhiều máy để dự đoán cùng lúc."
    )

    st.info(
        "Tệp CSV cần có đủ 6 cột: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], "
        "Torque [Nm], Tool wear [min]."
    )

    template = pd.DataFrame(columns=FEATURES)

    st.download_button(
        "⬇️ Tải file CSV mẫu",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="machine_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Chọn tệp CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            is_valid, validated_data = validate_input_data(
                input_df
            )

            if not is_valid:
                st.error(validated_data)

            else:
                predictions = model.predict(validated_data)

                probabilities = model.predict_proba(
                    validated_data
                )[:, 1]

                result_df = input_df.copy()

                result_df["Predicted Failure"] = predictions
                result_df["Failure Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                result_df["Trạng thái"] = [
                    "Có nguy cơ hỏng" if value == 1
                    else "Bình thường"
                    for value in predictions
                ]

                total = len(result_df)
                failure_count = int((predictions == 1).sum())
                normal_count = total - failure_count

                st.success(
                    f"Đã dự đoán thành công {total} bản ghi."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Tổng số máy", total)
                col2.metric("Bình thường", normal_count)
                col3.metric("Có nguy cơ hỏng", failure_count)

                st.subheader("Kết quả dự đoán")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "⬇️ Tải kết quả dự đoán",
                    data=csv_data,
                    file_name="machine_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                f"Không thể đọc hoặc xử lý tệp CSV: {error}"
            )


# =====================================================
# 8. SO SÁNH MÔ HÌNH
# =====================================================

elif page == "So sánh mô hình":

    st.title("Đánh giá và so sánh mô hình")

    st.write(
        "So sánh hiệu suất của năm thuật toán "
        "trên cùng tập dữ liệu kiểm thử."
    )

    display_comparison = comparison.copy()

    display_comparison["Model"] = (
        display_comparison["Model"].map(MODEL_LABELS)
        .fillna(display_comparison["Model"])
    )

    st.subheader("Bảng kết quả đánh giá")

    st.dataframe(
        display_comparison.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Biểu đồ so sánh")

    metric_options = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1-score": "F1-score",
        "ROC-AUC": "ROC-AUC"
    }

    selected_metric = st.selectbox(
        "Chọn chỉ số đánh giá",
        list(metric_options.keys())
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    chart_data = comparison.sort_values(
        by=selected_metric,
        ascending=True
    )

    chart_labels = [
        MODEL_LABELS.get(name, name)
        for name in chart_data["Model"]
    ]

    bars = ax.barh(
        chart_labels,
        chart_data[selected_metric]
    )

    ax.set_xlabel(selected_metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"So sánh mô hình theo {selected_metric}")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center"
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best_row = comparison.loc[
        comparison["F1-score"].idxmax()
    ]

    best_name = MODEL_LABELS.get(
        best_row["Model"],
        best_row["Model"]
    )

    st.success(
        f"Mô hình có F1-score cao nhất: {best_name} "
        f"({best_row['F1-score']:.4f})"
    )

    st.divider()

    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")

    try:
        cm_df = load_confusion_matrix()

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Confusion Matrix - Random Forest")
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")

        ax.set_xticklabels(
            ["Bình thường", "Hỏng hóc"]
        )
        ax.set_yticklabels(
            ["Bình thường", "Hỏng hóc"],
            rotation=0
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.warning(
            f"Không thể hiển thị Confusion Matrix: {error}"
        )


# =====================================================
# 9. CHÂN TRANG
# =====================================================

st.divider()

st.caption(
    "Machine Failure Prediction | "
    "Ứng dụng học máy dự đoán nguy cơ hỏng hóc máy móc"
)
