import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML trong Giáo dục",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
}
.main-hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.main-hero p { font-size: 1.05rem; opacity: 0.85; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

.algo-card {
    background: #f8faff;
    border: 1.5px solid #e2e8f4;
    border-left: 5px solid #3b7dd8;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.algo-card h4 { color: #1e3a5f; margin: 0 0 0.4rem 0; font-size: 1.05rem; font-weight: 600; }
.algo-card p  { color: #4a5568; font-size: 0.9rem; margin: 0; }

.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.2rem;
    color: white;
    text-align: center;
}
.metric-box .val { font-size: 2rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.metric-box .lbl { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 3px solid #3b7dd8;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}
.info-box {
    background: #eef4ff;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    border: 1px solid #c3d8f7;
    color: #1e3a5f;
    font-size: 0.93rem;
    line-height: 1.6;
}
.footer {
    text-align: center;
    color: #718096;
    font-size: 0.83rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 ML trong Giáo dục")
    st.markdown("---")
    page = st.radio(
        "Chọn nội dung",
        [
            "🏠 Tổng quan",
            "🌳 Decision Tree",
            "👥 K-Nearest Neighbors",
            "📈 Linear Regression",
            "🔵 Logistic Regression",
            "📊 Naive Bayes",
            "⚖️ So sánh thuật toán",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#718096; line-height:1.7'>
    <b>📚 Môn học:</b> Nhập môn KHDL<br>
    <b>🎯 Chủ đề:</b> Thuật toán ML<br>
    <b>📌 Lĩnh vực:</b> Ứng dụng Giáo dục
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Tổng quan":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">📖 NHẬP MÔN KHOA HỌC DỮ LIỆU</div>
        <h1>Thuật Toán Học Máy<br>& Ứng Dụng trong Giáo Dục</h1>
        <p>Khám phá 5 thuật toán Machine Learning phổ biến và cách chúng được ứng dụng
        để cải thiện chất lượng giáo dục — từ dự đoán kết quả học tập đến phân loại học sinh.</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["5", "3", "2", "∞"],
        ["Thuật toán", "Bài toán phân loại", "Bài toán hồi quy", "Khả năng ứng dụng"],
    ):
        col.markdown(f"""
        <div class="metric-box">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Các thuật toán được trình bày</div>', unsafe_allow_html=True)

    algos = [
        ("🌳 Decision Tree", "Cây quyết định", "Phân loại học sinh theo kết quả học tập, nhận diện các yếu tố ảnh hưởng đến điểm số."),
        ("👥 K-Nearest Neighbors", "K hàng xóm gần nhất", "Gợi ý tài liệu học tập dựa trên hồ sơ học sinh tương tự, phân nhóm học lực."),
        ("📈 Linear Regression", "Hồi quy tuyến tính", "Dự đoán điểm thi cuối kỳ từ điểm giữa kỳ, thời gian học, số buổi tham dự."),
        ("🔵 Logistic Regression", "Hồi quy Logistic", "Dự đoán nguy cơ học sinh bỏ học, phân loại đậu/rớt kỳ thi."),
        ("📊 Naive Bayes", "Bayes ngây thơ", "Phân loại câu hỏi trắc nghiệm theo chủ đề, phát hiện gian lận thi cử."),
    ]

    for icon_name, vn_name, app_desc in algos:
        st.markdown(f"""
        <div class="algo-card">
            <h4>{icon_name} — {vn_name}</h4>
            <p>🎓 <b>Ứng dụng:</b> {app_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔄 Quy trình học máy tổng quát</div>', unsafe_allow_html=True)

    steps = ["Thu thập\ndữ liệu", "Tiền xử lý\ndữ liệu", "Chọn\nmô hình", "Huấn\nluyện", "Đánh\ngiá", "Triển\nkhai"]
    colors = ["#3b7dd8", "#5a95e8", "#7ab0f0", "#9ac8f5", "#b4d9f8", "#cce8fb"]

    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (s, c) in enumerate(zip(steps, colors)):
        rect = mpatches.FancyBboxPatch((i + 0.08, 0.2), 0.75, 0.6,
                                        boxstyle="round,pad=0.04", color=c, zorder=2)
        ax.add_patch(rect)
        ax.text(i + 0.455, 0.515, s, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3, linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.93, 0.5), xytext=(i + 0.83, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#1e3a5f", lw=2))
    fig.patch.set_facecolor("#f8faff")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div class="info-box">
    💡 <b>Hướng dẫn sử dụng:</b> Chọn từng thuật toán ở thanh bên trái để xem lý thuyết,
    chạy thử trực tiếp trên dữ liệu giáo dục mô phỏng, và quan sát kết quả trực quan.
    Trang <b>So sánh thuật toán</b> sẽ đánh giá hiệu quả tổng hợp của cả 5 mô hình.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — synthetic education dataset
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_edu_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    study_hours   = rng.uniform(1, 10, n)
    attendance    = rng.uniform(50, 100, n)
    midterm       = rng.uniform(3, 10, n)
    sleep_hours   = rng.uniform(4, 9, n)
    activities    = rng.integers(0, 5, n)

    final = (
        2.5 * study_hours
        + 0.05 * attendance
        + 0.6 * midterm
        + 0.2 * sleep_hours
        - 0.1 * activities
        + rng.normal(0, 0.8, n)
    )
    final = np.clip(final, 0, 10)

    labels = pd.cut(final, bins=[0, 5, 6.5, 8, 10],
                    labels=["Yếu", "Trung bình", "Khá", "Giỏi"])

    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance":  attendance,
        "midterm":     midterm,
        "sleep_hours": sleep_hours,
        "activities":  activities,
        "final_score": final,
        "grade":       labels,
    })


def split_scale(df, features, target, test_size=0.2):
    X = df[features].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te


def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels or "auto",
                yticklabels=labels or "auto", ax=ax)
    ax.set_xlabel("Dự đoán", fontsize=10)
    ax.set_ylabel("Thực tế", fontsize=10)
    ax.set_title("Ma trận nhầm lẫn", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Decision Tree":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">THUẬT TOÁN 1 / 5</div>
        <h1>🌳 Decision Tree<br><span style='font-size:1.4rem;font-weight:500'>Cây Quyết Định</span></h1>
        <p>Phân loại học sinh theo xếp loại học lực dựa trên các chỉ số học tập.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📖 Lý thuyết</div>', unsafe_allow_html=True)
    with st.expander("Xem lý thuyết thuật toán", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **Decision Tree** chia dữ liệu bằng cách chọn đặc trưng tốt nhất tại mỗi nút theo tiêu chí:
        - **Gini Impurity**: đo mức độ "không thuần" của tập dữ liệu
        - **Information Gain / Entropy**: dựa trên lý thuyết thông tin

        **Ưu điểm:** Dễ giải thích, không cần chuẩn hóa dữ liệu, xử lý được cả phân loại lẫn số.

        **Nhược điểm:** Dễ overfit nếu không kiểm soát độ sâu cây.
        """)
        c2.markdown("""
        **Ứng dụng trong giáo dục:**
        - Phân loại học sinh: Yếu / Trung bình / Khá / Giỏi
        - Nhận diện nhân tố ảnh hưởng kết quả học tập
        - Hỗ trợ giáo viên ra quyết định can thiệp sớm
        - Xây dựng hệ thống tư vấn học bổng

        **Công thức Gini:**
        $$Gini = 1 - \\sum_{i=1}^{C} p_i^2$$
        """)

    st.markdown('<div class="section-title">🧪 Thực hành</div>', unsafe_allow_html=True)
    df = generate_edu_data()

    col_p, col_main = st.columns([1, 3])
    with col_p:
        max_depth = st.slider("Độ sâu tối đa (max_depth)", 2, 10, 4)
        criterion = st.selectbox("Tiêu chí phân chia", ["gini", "entropy"])
        features  = st.multiselect("Đặc trưng đầu vào",
                                   ["study_hours", "attendance", "midterm", "sleep_hours", "activities"],
                                   default=["study_hours", "attendance", "midterm"])

    if len(features) < 1:
        st.warning("Chọn ít nhất 1 đặc trưng.")
        st.stop()

    le = LabelEncoder()
    df["grade_enc"] = le.fit_transform(df["grade"])
    X_tr, X_te, y_tr, y_te = split_scale(df, features, "grade_enc")

    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    with col_main:
        m1, m2, m3 = st.columns(3)
        m1.metric("Độ chính xác", f"{acc:.2%}")
        m2.metric("Số nút lá", clf.get_n_leaves())
        m3.metric("Độ sâu thực tế", clf.get_depth())

        tab1, tab2, tab3 = st.tabs(["Hình cây", "Ma trận nhầm lẫn", "Tầm quan trọng đặc trưng"])

        with tab1:
            fig, ax = plt.subplots(figsize=(14, 5))
            plot_tree(clf, feature_names=features,
                      class_names=le.classes_, filled=True,
                      rounded=True, fontsize=8, ax=ax, max_depth=3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            st.pyplot(plot_confusion(y_te, y_pred, labels=list(range(len(le.classes_)))), use_container_width=True)
            plt.close()

        with tab3:
            imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            imp.plot.barh(ax=ax, color="#3b7dd8")
            ax.set_title("Feature Importance", fontweight="bold")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KNN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 K-Nearest Neighbors":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">THUẬT TOÁN 2 / 5</div>
        <h1>👥 K-Nearest Neighbors<br><span style='font-size:1.4rem;font-weight:500'>K Hàng Xóm Gần Nhất</span></h1>
        <p>Phân loại học sinh dựa trên sự tương đồng với các học sinh đã biết kết quả.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Lý thuyết", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **KNN** phân loại một điểm mới bằng cách tìm K điểm gần nhất trong tập huấn luyện và lấy nhãn phổ biến nhất.

        **Khoảng cách Euclidean:**
        $$d(x,y) = \\sqrt{\\sum_{i=1}^{n}(x_i - y_i)^2}$$

        **Ưu điểm:** Không cần huấn luyện (lazy learning), đơn giản, hiệu quả với dữ liệu nhỏ.

        **Nhược điểm:** Chậm với dữ liệu lớn, nhạy cảm với nhiễu và chiều dữ liệu cao.
        """)
        c2.markdown("""
        **Ứng dụng trong giáo dục:**
        - Gợi ý tài liệu học tập cho học sinh tương tự
        - Phân nhóm học sinh theo phong cách học
        - Tìm học sinh cần hỗ trợ thêm
        - Hệ thống gợi ý khóa học (recommender)
        """)

    df = generate_edu_data()
    le = LabelEncoder()
    df["grade_enc"] = le.fit_transform(df["grade"])
    features = ["study_hours", "attendance", "midterm"]

    col_p, col_main = st.columns([1, 3])
    with col_p:
        k = st.slider("Số hàng xóm K", 1, 20, 5)
        metric = st.selectbox("Khoảng cách", ["euclidean", "manhattan", "minkowski"])

    X_tr, X_te, y_tr, y_te = split_scale(df, features, "grade_enc")
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    with col_main:
        m1, m2 = st.columns(2)
        m1.metric("Độ chính xác", f"{acc:.2%}")
        m2.metric("K neighbors", k)

        tab1, tab2, tab3 = st.tabs(["Đường cong K", "Ma trận nhầm lẫn", "Dự đoán học sinh mới"])

        with tab1:
            k_vals = range(1, 21)
            accs = []
            for kv in k_vals:
                m = KNeighborsClassifier(n_neighbors=kv, metric=metric)
                m.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, m.predict(X_te)))
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(k_vals, accs, "o-", color="#3b7dd8", lw=2, markersize=5)
            ax.axvline(k, color="#e53e3e", ls="--", label=f"K={k}")
            ax.set_xlabel("K")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy theo K", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            st.pyplot(plot_confusion(y_te, y_pred), use_container_width=True)
            plt.close()

        with tab3:
            st.markdown("**Nhập thông tin học sinh mới:**")
            sh = st.slider("Giờ học/ngày", 1.0, 10.0, 5.0, 0.5)
            at = st.slider("Tỷ lệ tham dự (%)", 50, 100, 80)
            mt = st.slider("Điểm giữa kỳ", 0.0, 10.0, 6.5, 0.5)
            scaler = StandardScaler()
            scaler.fit(df[features].values)
            xnew = scaler.transform([[sh, at, mt]])
            pred_label = le.classes_[knn.predict(xnew)[0]]
            proba = knn.predict_proba(xnew)[0]
            st.success(f"🎯 Dự đoán xếp loại: **{pred_label}**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(le.classes_, proba, color="#3b7dd8")
            ax.set_xlabel("Xác suất")
            ax.set_title("Phân phối xác suất", fontsize=10, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Linear Regression":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">THUẬT TOÁN 3 / 5</div>
        <h1>📈 Linear Regression<br><span style='font-size:1.4rem;font-weight:500'>Hồi Quy Tuyến Tính</span></h1>
        <p>Dự đoán điểm thi cuối kỳ của học sinh từ các chỉ số học tập.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Lý thuyết", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **Linear Regression** mô hình hóa quan hệ tuyến tính giữa biến phụ thuộc y và các biến độc lập x:

        $$\\hat{y} = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n$$

        **Hàm mất mát (MSE):**
        $$MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

        Tham số tối ưu bằng **Least Squares:** $\\beta = (X^TX)^{-1}X^Ty$
        """)
        c2.markdown("""
        **Ứng dụng trong giáo dục:**
        - Dự đoán điểm thi cuối kỳ từ điểm giữa kỳ
        - Ước lượng thời gian cần thiết để đạt điểm mục tiêu
        - Phân tích tác động của số buổi học đến kết quả
        - Lập kế hoạch học tập cá nhân hóa

        **Đánh giá mô hình:**
        - R² (coefficient of determination): càng gần 1 càng tốt
        - RMSE: sai số trung bình bình phương
        """)

    df = generate_edu_data()
    col_p, col_main = st.columns([1, 3])
    with col_p:
        feature_x = st.selectbox("Biến đầu vào chính (trục x)",
                                  ["study_hours", "attendance", "midterm", "sleep_hours"])
        use_multi = st.checkbox("Dùng hồi quy bội (multiple)", value=False)

    if use_multi:
        features = ["study_hours", "attendance", "midterm", "sleep_hours"]
    else:
        features = [feature_x]

    X = df[features].values
    y = df["final_score"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    with col_main:
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{r2:.4f}")
        m2.metric("RMSE", f"{np.sqrt(mse):.3f}")
        m3.metric("MSE", f"{mse:.3f}")

        if not use_multi:
            tab1, tab2 = st.tabs(["Scatter + đường hồi quy", "Phân phối sai số"])
            with tab1:
                xv = df[feature_x].values
                xline = np.linspace(xv.min(), xv.max(), 100).reshape(-1, 1)
                yline = reg.predict(xline)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(df[feature_x], df["final_score"], alpha=0.4, color="#3b7dd8", s=25, label="Thực tế")
                ax.plot(xline, yline, color="#e53e3e", lw=2.5, label="Hồi quy")
                ax.set_xlabel(feature_x, fontsize=10)
                ax.set_ylabel("Điểm cuối kỳ", fontsize=10)
                ax.set_title(f"Hồi quy: {feature_x} → final_score", fontweight="bold")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with tab2:
                resid = y_te - y_pred
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.hist(resid, bins=20, color="#3b7dd8", edgecolor="white", alpha=0.85)
                ax.axvline(0, color="#e53e3e", lw=2, ls="--")
                ax.set_xlabel("Sai số (residual)", fontsize=10)
                ax.set_title("Phân phối sai số", fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_te, y_pred, alpha=0.5, color="#3b7dd8", s=30)
            mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Dự đoán hoàn hảo")
            ax.set_xlabel("Thực tế", fontsize=10)
            ax.set_ylabel("Dự đoán", fontsize=10)
            ax.set_title("Thực tế vs Dự đoán (Multiple Regression)", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("**Hệ số hồi quy:**")
            coef_df = pd.DataFrame({"Đặc trưng": features, "Hệ số": reg.coef_})
            st.dataframe(coef_df.set_index("Đặc trưng"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Logistic Regression":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">THUẬT TOÁN 4 / 5</div>
        <h1>🔵 Logistic Regression<br><span style='font-size:1.4rem;font-weight:500'>Hồi Quy Logistic</span></h1>
        <p>Dự đoán xác suất đậu/rớt kỳ thi và phát hiện học sinh có nguy cơ.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Lý thuyết", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **Logistic Regression** dùng hàm sigmoid để chuyển đổi đầu ra về xác suất [0,1]:

        $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

        $$P(y=1|x) = \\sigma(\\beta^T x)$$

        **Hàm mất mát (Binary Cross-Entropy):**
        $$L = -\\frac{1}{n}\\sum[y\\log\\hat{p} + (1-y)\\log(1-\\hat{p})]$$
        """)
        c2.markdown("""
        **Ứng dụng trong giáo dục:**
        - Dự đoán xác suất học sinh đậu/rớt
        - Phát hiện sớm học sinh có nguy cơ bỏ học
        - Phân loại học sinh cần hỗ trợ tâm lý
        - Tư vấn hướng nghiệp dựa trên học lực

        **Lưu ý:** Logistic Regression cho xác suất, không chỉ nhãn,
        nên rất hữu ích để quyết định mức độ can thiệp.
        """)

    df = generate_edu_data()
    df["pass"] = (df["final_score"] >= 5).astype(int)
    features = ["study_hours", "attendance", "midterm", "sleep_hours"]

    col_p, col_main = st.columns([1, 3])
    with col_p:
        C = st.select_slider("Regularization C", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        threshold = st.slider("Ngưỡng quyết định", 0.1, 0.9, 0.5, 0.05)

    X_tr, X_te, y_tr, y_te = split_scale(df, features, "pass")
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_te, y_pred)

    with col_main:
        m1, m2, m3 = st.columns(3)
        m1.metric("Độ chính xác", f"{acc:.2%}")
        n_risk = (y_proba < threshold).sum()
        m2.metric("HS nguy cơ rớt", n_risk)
        m3.metric("Ngưỡng", threshold)

        tab1, tab2, tab3 = st.tabs(["Phân phối xác suất", "Ma trận nhầm lẫn", "Sigmoid curve"])

        with tab1:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist([y_proba[y_te == 0], y_proba[y_te == 1]],
                    bins=20, label=["Rớt (thực tế)", "Đậu (thực tế)"],
                    color=["#e53e3e", "#38a169"], alpha=0.75, stacked=False)
            ax.axvline(threshold, color="#d69e2e", ls="--", lw=2, label=f"Ngưỡng={threshold}")
            ax.set_xlabel("Xác suất đậu", fontsize=10)
            ax.set_ylabel("Số học sinh", fontsize=10)
            ax.set_title("Phân phối xác suất theo kết quả thực", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            st.pyplot(plot_confusion(y_te, y_pred, labels=[0, 1]), use_container_width=True)
            plt.close()

        with tab3:
            z = np.linspace(-8, 8, 300)
            sig = 1 / (1 + np.exp(-z))
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(z, sig, color="#3b7dd8", lw=2.5)
            ax.axhline(threshold, color="#e53e3e", ls="--", label=f"Ngưỡng={threshold}")
            ax.axhline(0.5, color="#718096", ls=":", lw=1)
            ax.fill_between(z, sig, threshold, where=(sig < threshold), alpha=0.12, color="#e53e3e", label="Vùng rớt")
            ax.fill_between(z, sig, threshold, where=(sig >= threshold), alpha=0.12, color="#38a169", label="Vùng đậu")
            ax.set_xlabel("z = β·x", fontsize=10)
            ax.set_ylabel("σ(z)", fontsize=10)
            ax.set_title("Hàm Sigmoid", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Naive Bayes":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">THUẬT TOÁN 5 / 5</div>
        <h1>📊 Naive Bayes<br><span style='font-size:1.4rem;font-weight:500'>Phân loại Bayes Ngây Thơ</span></h1>
        <p>Phân loại xếp loại học lực dựa trên định lý Bayes.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Lý thuyết", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **Naive Bayes** áp dụng định lý Bayes với giả định các đặc trưng **độc lập có điều kiện**:

        $$P(C|X) = \\frac{P(X|C) \\cdot P(C)}{P(X)}$$

        **Gaussian Naive Bayes** giả định mỗi đặc trưng phân phối chuẩn:
        $$P(x_i|C) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x_i-\\mu)^2}{2\\sigma^2}}$$

        **Ưu điểm:** Rất nhanh, tốt với dữ liệu nhỏ, hiệu quả với text.
        """)
        c2.markdown("""
        **Ứng dụng trong giáo dục:**
        - Phân loại câu hỏi theo chủ đề (NLP)
        - Phát hiện bài làm có dấu hiệu gian lận
        - Lọc nội dung không phù hợp trong hệ thống LMS
        - Xếp loại học sinh từ nhiều chỉ số khác nhau
        - Hệ thống hỏi đáp tự động (FAQ bot)
        """)

    df = generate_edu_data()
    le = LabelEncoder()
    df["grade_enc"] = le.fit_transform(df["grade"])
    features = ["study_hours", "attendance", "midterm", "sleep_hours", "activities"]

    col_p, col_main = st.columns([1, 3])
    with col_p:
        var_smoothing = st.select_slider("Var smoothing (log10)",
                                          options=[-12, -10, -9, -8, -6, -4], value=-9)

    X_tr, X_te, y_tr, y_te = split_scale(df, features, "grade_enc")
    nb = GaussianNB(var_smoothing=10 ** var_smoothing)
    nb.fit(X_tr, y_tr)
    y_pred  = nb.predict(X_te)
    y_proba = nb.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)

    with col_main:
        m1, m2, m3 = st.columns(3)
        m1.metric("Độ chính xác", f"{acc:.2%}")
        m2.metric("Số lớp", len(le.classes_))
        m3.metric("Mẫu huấn luyện", len(X_tr))

        tab1, tab2, tab3 = st.tabs(["Ma trận nhầm lẫn", "Phân phối xác suất", "Class statistics"])

        with tab1:
            st.pyplot(plot_confusion(y_te, y_pred), use_container_width=True)
            plt.close()

        with tab2:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=False)
            for i, (ax, cls) in enumerate(zip(axes, le.classes_)):
                ax.hist(y_proba[:, i], bins=15, color="#3b7dd8", edgecolor="white", alpha=0.85)
                ax.set_title(cls, fontsize=9, fontweight="bold")
                ax.set_xlabel("P", fontsize=8)
            fig.suptitle("Phân phối xác suất theo lớp", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab3:
            rows = []
            for i, cls in enumerate(le.classes_):
                rows.append({
                    "Lớp": cls,
                    "Số mẫu": (y_tr == i).sum(),
                    "Prior P(C)": f"{(y_tr == i).mean():.3f}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Lớp"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SO SÁNH THUẬT TOÁN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ So sánh thuật toán":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">TỔNG HỢP & ĐÁNH GIÁ</div>
        <h1>⚖️ So Sánh Thuật Toán<br><span style='font-size:1.4rem;font-weight:500'>Benchmark trên dữ liệu giáo dục</span></h1>
        <p>Đánh giá và so sánh hiệu suất của 5 thuật toán trên cùng một tập dữ liệu.</p>
    </div>
    """, unsafe_allow_html=True)

    df = generate_edu_data()
    le = LabelEncoder()
    df["grade_enc"] = le.fit_transform(df["grade"])
    features = ["study_hours", "attendance", "midterm", "sleep_hours", "activities"]

    X_tr, X_te, y_tr, y_te = split_scale(df, features, "grade_enc")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN (K=7)":     KNeighborsClassifier(n_neighbors=7),
        "Logistic Reg.": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":   GaussianNB(),
    }

    results = []
    preds   = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        yp  = model.predict(X_te)
        acc = accuracy_score(y_te, yp)
        preds[name] = yp
        results.append({"Thuật toán": name, "Accuracy": acc})

    res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    st.markdown('<div class="section-title">📊 Bảng xếp hạng</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 2])
    with c1:
        st.dataframe(
            res_df.style.format({"Accuracy": "{:.2%}"})
                        .background_gradient(subset=["Accuracy"], cmap="Blues"),
            use_container_width=True,
        )
    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors_bar = ["#2d6a4f", "#3b7dd8", "#764ba2", "#e53e3e"]
        bars = ax.barh(res_df["Thuật toán"], res_df["Accuracy"], color=colors_bar[::-1], height=0.55)
        for bar, val in zip(bars, res_df["Accuracy"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2%}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Accuracy", fontsize=10)
        ax.set_title("So sánh Accuracy", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">🗺️ Ma trận nhầm lẫn — Tất cả mô hình</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, yp) in zip(axes, preds.items()):
        cm = confusion_matrix(y_te, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Dự đoán", fontsize=8)
        ax.set_ylabel("Thực tế", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">📋 Nhận xét tổng hợp</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <table style="width:100%; border-collapse:collapse; font-size:0.9rem">
    <tr style="background:#dbeafe">
        <th style="padding:8px;text-align:left">Thuật toán</th>
        <th>Ưu điểm</th>
        <th>Nhược điểm</th>
        <th>Phù hợp nhất khi</th>
    </tr>
    <tr><td><b>Decision Tree</b></td><td>Dễ giải thích, trực quan</td><td>Dễ overfit</td><td>Cần giải thích quyết định cho GV</td></tr>
    <tr style="background:#f8faff"><td><b>KNN</b></td><td>Đơn giản, không cần train</td><td>Chậm khi dữ liệu lớn</td><td>Gợi ý cá nhân hóa</td></tr>
    <tr><td><b>Logistic Reg.</b></td><td>Cho xác suất, ổn định</td><td>Giả định tuyến tính</td><td>Dự đoán rủi ro bỏ học</td></tr>
    <tr style="background:#f8faff"><td><b>Naive Bayes</b></td><td>Rất nhanh, ít dữ liệu</td><td>Giả định độc lập</td><td>Phân loại văn bản, câu hỏi</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:1rem">
    🏆 <b>Kết luận:</b> Không có thuật toán "tốt nhất" tuyệt đối. Việc lựa chọn phụ thuộc vào:
    kích thước dữ liệu, yêu cầu giải thích mô hình, tài nguyên tính toán, và bài toán cụ thể trong giáo dục.
    Trong thực tế, nên thử nhiều mô hình và so sánh trên dữ liệu thực của trường/lớp.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🎓 Đề tài: <b>Tìm hiểu về một số thuật toán học máy và ứng dụng trong giáo dục</b><br>
    Môn: Nhập môn Khoa học Dữ liệu &nbsp;|&nbsp; Xây dựng bằng Python · Streamlit · Scikit-learn
</div>
""", unsafe_allow_html=True)
