# 🎓 Thuật Toán Học Máy & Ứng Dụng trong Giáo Dục

> **Môn học:** Nhập môn Khoa học Dữ liệu  
> **Chủ đề:** Tìm hiểu về một số thuật toán học máy và ứng dụng trong giáo dục

Ứng dụng Streamlit tương tác giúp khám phá **5 thuật toán Machine Learning** phổ biến thông qua bài toán giáo dục thực tế — phân loại học lực, dự đoán điểm thi, phát hiện học sinh có nguy cơ.

---

## 📂 Cấu trúc dự án

```
TieuLuanKHDL.
├── app.py            # Toàn bộ ứng dụng Streamlit
├── requirements.txt  # Danh sách thư viện cần thiết
└── README.md         # Tài liệu hướng dẫn (file này)
```

---

## 🧠 Nội dung thuật toán

| Trang | Thuật toán | Bài toán giáo dục |
|-------|-----------|-------------------|
| 🌳 Decision Tree | Cây quyết định | Phân loại xếp loại học lực (Yếu/TB/Khá/Giỏi) |
| 👥 KNN | K Hàng xóm gần nhất | Gợi ý tài liệu & phân nhóm học sinh |
| 📈 Linear Regression | Hồi quy tuyến tính | Dự đoán điểm thi cuối kỳ |
| 🔵 Logistic Regression | Hồi quy Logistic | Dự đoán xác suất đậu/rớt |
| 📊 Naive Bayes | Bayes ngây thơ | Phân loại câu hỏi, phát hiện gian lận |
| ⚖️ So sánh | Benchmark | Đánh giá tổng hợp 5 mô hình |

---

## 📊 Dữ liệu mô phỏng

Ứng dụng tự sinh dữ liệu học sinh với các đặc trưng:

| Đặc trưng | Mô tả | Khoảng giá trị |
|-----------|-------|----------------|
| `study_hours` | Số giờ tự học mỗi ngày | 1 – 10 giờ |
| `attendance` | Tỷ lệ tham dự lớp | 50 – 100% |
| `midterm` | Điểm thi giữa kỳ | 0 – 10 |
| `sleep_hours` | Số giờ ngủ mỗi ngày | 4 – 9 giờ |
| `activities` | Số hoạt động ngoại khóa | 0 – 4 |
| `final_score` | Điểm thi cuối kỳ *(nhãn)* | 0 – 10 |
| `grade` | Xếp loại *(nhãn)* | Yếu / TB / Khá / Giỏi |

---

## 🛠️ Công nghệ sử dụng

- **[Streamlit](https://streamlit.io/)** — Giao diện web tương tác
- **[Scikit-learn](https://scikit-learn.org/)** — Thuật toán Machine Learning
- **[Pandas](https://pandas.pydata.org/)** — Xử lý dữ liệu
- **[NumPy](https://numpy.org/)** — Tính toán số học
- **[Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/)** — Trực quan hóa

---

## ✨ Tính năng nổi bật

- **Lý thuyết + Demo tích hợp** — Mỗi thuật toán có phần lý thuyết và demo chạy thực luôn
- **Tham số tùy chỉnh** — Điều chỉnh siêu tham số và thấy kết quả thay đổi ngay lập tức
- **Dự đoán học sinh mới** — Nhập thông tin học sinh và xem dự đoán (trang KNN)
- **So sánh toàn diện** — Benchmark 5 mô hình, bảng xếp hạng, confusion matrix song song
- **Hoàn toàn offline** — Không cần internet sau khi cài đặt, dữ liệu tự sinh

---

## 📚 Tài liệu tham khảo

1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
2. Baker, R.S. & Inventado, P.S. (2014). Educational Data Mining and Learning Analytics. *Learning and Knowledge Analytics*.
3. Romero, C. & Ventura, S. (2010). Educational data mining: A review of the state of the art. *IEEE Transactions on Systems, Man, and Cybernetics*.
4. Scikit-learn documentation: https://scikit-learn.org/stable/
5. Streamlit documentation: https://docs.streamlit.io/

---

<div align="center">
Lê Văn Thành - 24T1110009
</div>
