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
# HELPER FUNCTIONS (Đã chuyển lên đầu để tránh ngắt mạch if-elif)
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
                      rounded=True, fontsize=8, ax=
