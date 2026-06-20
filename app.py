import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hệ Thống Học Máy Hỗ Trợ Giáo Dục",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS (Giao diện hiện đại, trực quan) ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
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
}
.main-hero p { font-size: 1.1rem; opacity: 0.9; margin: 0; }

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-card .num { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; }
.metric-card .label { font-size: 0.85rem; color: #4b5563; font-weight: 500; margin-top: 4px; }

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1rem 0;
}

.edu-box {
    background: #f0fdf4;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 1.2rem;
    margin-top: 1rem;
}
.edu-box-danger {
    background: #fef2f2;
    border-left: 5px solid #dc2626;
    border-radius: 8px;
    padding: 1.2rem;
    margin-top: 1rem;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    padding: 2rem 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)

# ── DATA GENERATOR (Mô phỏng dữ liệu học sinh thực tế) ─────────────────────────
@st.cache_data
def load_student_dataset(n_samples=400, seed=42):
    rng = np.random.default_rng(seed)
    study_hours   = rng.uniform(1.5, 10, n_samples)      # Giờ tự học / tuần
    attendance    = rng.uniform(60, 100, n_samples)      # % số tiết tham gia lớp
    midterm_score = rng.uniform(2.0, 10, n_samples)      # Điểm thi giữa kỳ
    digital_clicks= rng.uniform(10, 250, n_samples)      # Số lượt tương tác trên LMS (E-learning)
    
    # Tính toán điểm cuối kỳ dựa trên các trọng số giáo dục thực tế + nhiễu ngẫu nhiên
    final_score = (
        0.25 * study_hours 
        + 0.04 * attendance 
        + 0.45 * midterm_score 
        + 0.005 * digital_clicks 
        + rng.normal(0, 0.6, n_samples)
    )
    final_score = np.clip(final_score, 0, 10)
    
    # Gán nhãn học lực theo quy chế đào tạo
    grade = pd.cut(final_score, bins=[0, 5.0, 6.5, 8.0, 10.0],
                    labels=["Yếu/Kém", "Trung bình", "Khá", "Giỏi/Xuất sắc"])
    
    # Gán nhãn nguy cơ bỏ học (Dưới 4.5 điểm giữa kỳ và chuyên cần dưới 75%)
    dropout_risk = ((midterm_score < 4.5) & (attendance < 75)).astype(int)
    
    return pd.DataFrame({
        "Giờ tự học": study_hours,
        "Tỷ lệ chuyên cần": attendance,
        "Điểm giữa kỳ": midterm_score,
        "Tương tác LMS": digital_clicks,
        "Điểm cuối kỳ": final_score,
        "Xếp loại học lực": grade,
        "Nguy cơ bỏ học": dropout_risk
    })

df_edu = load_student_dataset()

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Quản Lý Học Đường")
    st.markdown("---")
    page = st.radio(
        "Lựa chọn phân hệ:",
        [
            "🏠 Trung tâm Điều hành (Dashboard)",
            "📈 Dự báo Điểm số (Hồi quy Tuyến tính)",
            "🌳 Phân loại Học lực (Cây Quyết định)",
            "👥 Gợi ý Lộ trình (K-Nhóm Hàng xóm)",
            "🚨 Cảnh báo Sớm Học sinh Bỏ học (Logistic)",
            "📊 Sàng lọc Đơn xin Học bổng (Naive Bayes)",
            "⚖️ Khảo sát Hiệu năng Mô hình"
        ]
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#4b5563;'>
    <b>Đề tài:</b> Ứng dụng KHDL & ML nâng cao chất lượng dạy và học.<br>
    <b>Dữ liệu:</b> 400 học sinh mô phỏng theo hành vi thực tế.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRUNG TÂM ĐIỀU HÀNH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Trung tâm Điều hành (Dashboard)":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Hệ Thống Trí Tuệ Nhân Tạo Học Đường</div>
        <h1>AI Trung Tâm Điều Hành Giáo Dục</h1>
        <p>Phân tích dữ liệu học tập tổng thể nhằm tối ưu hóa phương pháp giảng dạy và hỗ trợ can thiệp sư phạm kịp thời.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tổng quan chỉ số trắc lượng trường học
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="num">400</div><div class="label">Học sinh đang theo dõi</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="num">{df_edu["Điểm cuối kỳ"].mean():.2f}</div><div class="label">Điểm cuối kỳ TB</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="num">{df_edu["Tỷ lệ chuyên cần"].mean():.1f}%</div><div class="label">Tỷ lệ đi học chuyên cần</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="num" style="color:#dc2626">{df_edu["Nguy cơ bỏ học"].sum()}</div><div class="label">Học sinh cần trợ giúp gấp</div></div>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-title">📊 Bức tranh Toàn cảnh Học lực & Tương tác</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.countplot(data=df_edu, x="Xếp loại học lực", palette="Blues_r", ax=ax)
        ax.set_title("Phân phối Xếp loại Học lực Toàn trường", fontweight="bold", color="#1e3a8a")
        ax.set_ylabel("Số lượng học sinh")
        ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
    with col_g2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.scatterplot(data=df_edu, x="Giờ tự học", y="Điểm cuối kỳ", hue="Xếp loại học lực", palette="viridis", alpha=0.8, ax=ax)
        ax.set_title("Mối tương quan: Thời gian tự học vs Kết quả cuối kỳ", fontweight="bold", color="#1e3a8a")
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DỰ BÁO ĐIỂM SỐ (LINEAR REGRESSION)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Dự báo Điểm số (Hồi quy Tuyến tính)":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Bài toán Hồi Quy (Regression)</div>
        <h1>📈 Dự Đoán Điểm Thi Cuối Kỳ</h1>
        <p>Sử dụng thuật toán <b>Linear Regression</b> để ước lượng chính xác điểm số cuối kỳ của học sinh dựa trên quá trình học tập thực tế.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Tương tác LMS"]
    X = df_edu[features].values
    y = df_edu["Điểm cuối kỳ"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("### 🎛️ Giả lập học sinh mới")
        sh = st.slider("Giờ tự học/tuần", 0.0, 12.0, 5.0)
        at = st.slider("Chuyên cần (%)", 50, 100, 85)
        mt = st.slider("Điểm giữa kỳ", 0.0, 10.0, 6.0)
        lm = st.slider("Tương tác LMS (Click)", 0, 300, 120)
        
        pred_single = model.predict([[sh, at, mt, lm]])[0]
        pred_single = np.clip(pred_single, 0, 10)
        
        st.markdown(f"""
        <div class="edu-box">
            <h4>🎯 Kết quả dự báo AI:</h4>
            <span style="font-size:2rem; font-weight:bold; color:#16a34a">{pred_single:.2f} / 10</span> Điểm
            <p style="margin-top:5px; font-size:0.9rem; color:#374151"><b>Khuyến nghị:</b> Học sinh có xu hướng đạt kết quả an toàn. Giáo viên có thể giao thêm bài tập nâng cao để kích thích tư duy.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r:
        st.markdown("### 📊 Đánh giá chất lượng dự báo từ mô hình")
        m1, m2 = st.columns(2)
        m1.metric("Mức độ giải thích (R² Score)", f"{r2_score(y_test, y_pred):.2%}")
        m2.metric("Sai số trung bình (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f} Điểm")
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(y_test, y_pred, color="#3b82f6", alpha=0.6, edgecolors="w")
        ax.plot([0, 10], [0, 10], "r--", lw=2, label="Đường dự báo chuẩn")
        ax.set_xlabel("Điểm thực tế")
        ax.set_ylabel("Điểm AI dự báo")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. PHÂN LOẠI HỌC LỰC (DECISION TREE)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Phân loại Học lực (Cây Quyết định)":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Bài toán Phân Loại (Classification)</div>
        <h1>🌳 Phân Loại Học Lực Học Sinh</h1>
        <p>Mô hình <b>Decision Tree</b> bóc tách các điều kiện logic để tự động xếp lớp học lực (Giỏi, Khá, Trung bình, Yếu).</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Tương tác LMS"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_edu["Xếp loại học lực"])
    
    X_train, X_test, y_train, y_test = train_test_split(df_edu[features].values, y_encoded, test_size=0.2, random_state=42)
    
    depth = st.sidebar.slider("Độ sâu của cây quyết định", 2, 6, 3)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    st.markdown("### 🗺️ Sơ đồ cây quyết định phân loại (Trực quan hóa tiêu chí)")
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(clf, feature_names=features, class_names=le.classes_, filled=True, rounded=True, fontsize=9, ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.markdown(f"""
    <div class="edu-box">
        💡 <b>Ý nghĩa Sư phạm:</b> Nhìn vào sơ đồ cây, nhà quản lý giáo dục có thể thấy ngay "Nút thắt" đầu tiên chia rẽ học lực lớn nhất chính là <b>{features[clf.tree_.feature[0]]}</b>. Điều này chứng minh hành vi này quyết định cốt lõi đến chất lượng đầu ra của học sinh tại trường.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. GỢI Ý LỘ TRÌNH (KNN)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Gợi ý Lộ trình (K-Nhóm Hàng xóm)":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Hệ thống Khuyến Nghị (Recommender System)</div>
        <h1>👥 Gợi Ý Lộ Trình Học Tập Cá Nhân Hóa</h1>
        <p>Sử dụng <b>K-Nearest Neighbors</b> để tìm kiếm nhóm học sinh có hành vi tương đồng, từ đó gợi ý phương án ôn tập thích hợp.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ"]
    X = df_edu[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, df_edu["Xếp loại học lực"])
    
    st.markdown("### 🛠️ Phân tích hồ sơ một học sinh")
    c1, c2, c3 = st.columns(3)
    sh = c1.slider("Giờ tự học/tuần", 0.0, 10.0, 3.5)
    at = c2.slider("Tỷ lệ đi học (%)", 50, 100, 70)
    mt = c3.slider("Điểm thi giữa kỳ", 0.0, 10.0, 4.0)
    
    student_sample = scaler.transform([[sh, at, mt]])
    distances, indices = knn.kneighbors(student_sample)
    
    st.markdown("### 🤝 Bạn học có cùng xuất phát điểm (5 vị trí tương đồng nhất)")
    matched_students = df_edu.iloc[indices[0]]
    st.dataframe(matched_students[["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Điểm cuối kỳ", "Xếp loại học lực"]], use_container_width=True)
    
    avg_final_peers = matched_students["Điểm cuối kỳ"].mean()
    
    st.markdown(f"""
    <div class="edu-box-danger">
        ⚠️ <b>Chiến lược can thiệp từ AI:</b> 5 học sinh có hồ sơ tương tự bạn này trong quá khứ có điểm cuối kỳ trung bình là <b>{avg_final_peers:.2f}</b> và đa số rơi vào diện học lực trung bình yếu. <br>
        <b>👉 Khuyên dùng:</b> Nhà trường cần xếp học sinh này vào nhóm "Phụ đạo tích cực", tăng cường thời gian tự học lên tối thiểu 6 giờ/tuần để cải thiện tình hình.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CẢNH BÁO SỚM HỌC SINH BỎ HỌC (LOGISTIC REGRESSION)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Cảnh báo Sớm Học sinh Bỏ học (Logistic)":
    st.markdown("""
    <div class="main-hero" style="background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);">
        <div class="badge" style="background:rgba(0,0,0,0.3)">Hệ Thống Phòng Chống Rủi Rõ Học Đường</div>
        <h1>🚨 Cảnh Báo Sớm Nguy Cơ Bỏ Học</h1>
        <p>Thuật toán xác suất <b>Logistic Regression</b> phân tích dữ liệu chuyên cần và điểm số để tính toán xác suất (%) rủi ro một học sinh có khả năng nghỉ học giữa chừng.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Tương tác LMS"]
    X = df_edu[features].values
    y = df_edu["Nguy cơ bỏ học"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf_log = LogisticRegression()
    clf_log.fit(X_scaled, y)
    
    st.markdown("### 🔎 Kiểm tra mức độ rủi ro của học sinh nghi vấn")
    col1, col2 = st.columns(2)
    with col1:
        v_at = st.slider("Tỷ lệ chuyên cần hiện tại của học sinh (%)", 40, 100, 65)
        v_mt = st.slider("Điểm số giữa kỳ hiện tại", 0.0, 10.0, 3.5)
        v_sh = 2.0
        v_lms = 40
        
        test_student = scaler.transform([[v_sh, v_at, v_mt, v_lms]])
        risk_prob = clf_log.predict_proba(test_student)[0][1]
        
    with col2:
        st.markdown("#### Trạng thái cảnh báo từ hệ thống AI:")
        if risk_prob > 0.5:
            st.markdown(f"""
            <div class="edu-box-danger" style="text-align:center">
                <span style="font-size:1.2rem; font-weight:bold; color:#dc2626">🔴 BÁO ĐỘNG ĐỎ</span><br>
                <span style="font-size:3rem; font-weight:bold; color:#dc2626">{risk_prob:.1%}</span><br>
                <b>Nguy cơ bỏ học rất cao!</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="edu-box" style="text-align:center">
                <span style="font-size:1.2rem; font-weight:bold; color:#16a34a">🟢 AN TOÀN</span><br>
                <span style="font-size:3rem; font-weight:bold; color:#16a34a">{risk_prob:.1%}</span><br>
                <b>Nằm trong tầm kiểm soát.</b>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SÀNG LỌC ĐƠN XIN HỌC BỔNG (NAIVE BAYES)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sàng lọc Đơn xin Học bổng (Naive Bayes)":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Xử lý tự động hóa</div>
        <h1>📊 Sàng Lọc Học Bổng Tự Động</h1>
        <p>Ứng dụng định lý xác suất <b>Naive Bayes</b> để dự đoán nhanh khả năng đạt danh hiệu Giỏi/Xuất sắc dựa trên các điều kiện độc lập có sẵn.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Tương tác LMS"]
    X = df_edu[features].values
    y = (df_edu["Xếp loại học lực"] == "Giỏi/Xuất sắc").astype(int)
    
    nb = GaussianNB()
    nb.fit(X, y)
    
    st.markdown("### 📑 Thẩm định nhanh thông số xét duyệt hồ sơ")
    c1, c2 = st.columns(2)
    in_sh = c1.number_input("Số giờ tự học bình quân/tuần", 0.0, 15.0, 8.5)
    in_mt = c2.number_input("Điểm thi khảo sát giữa kỳ", 0.0, 10.0, 8.2)
    
    prob_scholarship = nb.predict_proba([[in_sh, 95.0, in_mt, 200.0]])[0][1]
    
    st.markdown(f"""
    <div class="edu-box">
        🎯 Xác suất học sinh này đạt danh hiệu học lực Giỏi/Xuất sắc cuối kỳ là: <b>{prob_scholarship:.2%}</b>.<br>
        Mô hình Bayes giúp bộ phận khảo thí đưa ra bộ lọc sơ bộ hồ sơ trước khi thành lập hội đồng chấm xét học bổng chính thức.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. KHẢO SÁT HIỆU NĂNG MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Khảo sát Hiệu năng Mô hình":
    st.markdown("""
    <div class="main-hero">
        <div class="badge">Đánh giá Khoa học Dữ liệu</div>
        <h1>⚖️ So Sánh Hiệu Năng 3 Thuật Toán Phân Loại</h1>
        <p>Báo cáo kỹ thuật so sánh độ chính xác của các thuật toán trên bài toán phân loại học lực học sinh.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ["Giờ tự học", "Tỷ lệ chuyên cần", "Điểm giữa kỳ", "Tương tác LMS"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_edu["Xếp loại học lực"])
    
    X_train, X_test, y_train, y_test = train_test_split(df_edu[features].values, y_encoded, test_size=0.2, random_state=42)
    
    models = {
        "Cây Quyết định (Decision Tree)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "K-Hàng xóm gần nhất (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Phân loại Bayes Ngây thơ (Naive Bayes)": GaussianNB()
    }
    
    scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores.append({"Mô hình học máy": name, "Độ chính xác (Accuracy)": acc})
        
    st.dataframe(pd.DataFrame(scores).style.format({"Độ chính xác (Accuracy)": "{:.2%}"}), use_container_width=True)
    
    st.markdown("""
    <div class="edu-box">
        💡 <b>Tổng kết:</b> Trong các bài toán quản lý giáo dục, ngoài <b>Độ chính xác (Accuracy)</b>, chúng ta cần ưu tiên chọn các mô hình có tính <b>Giải thích được (Explainable AI)</b> như Cây quyết định để giải trình rõ ràng lý do hệ thống đưa ra quyết định hoặc cảnh báo đối với học sinh cho phụ huynh và nhà trường.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🎓 Hệ thống mô phỏng KHDL ứng dụng Giáo dục | Xây dựng hoàn toàn bằng Thư viện Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
