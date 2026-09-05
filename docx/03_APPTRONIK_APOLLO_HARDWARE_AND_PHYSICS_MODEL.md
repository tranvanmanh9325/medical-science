# 🤖 TÀI LIỆU KỸ THUẬT 03: ĐẶC TẢ MÔ HÌNH VẬT LÝ & PHẦN CỨNG ROBOT APPTRONIK APOLLO

> **Dự án**: Apptronik Apollo Humanoid Robotics (`medical-science`)  
> **Chuyên đề**: Cấu Trúc Mô Hình MuJoCo MJCF, Phân Đoạn Cơ Thể, Hệ Thống Động Cơ & Cảm Biến  
> **Mã tài liệu**: `DOCX-HW-03` | **Phiên bản**: 2.4.0

---

## 📑 MỤC LỤC
1. [Nguồn gốc & Tiêu chuẩn Mô hình Google DeepMind Menagerie](#1-nguồn-gốc--tiêu-chuẩn-mô-hình)
2. [Cấu trúc Cây Động học & Danh mục 37 Phân đoạn Thân (Bodies Registry)](#2-cấu-trúc-cây-động-học--danh-mục-37-phân-đoạn-thân)
3. [Đặc tính Kỹ thuật 32 Động cơ Servo & Tỷ số Truyền](#3-đặc-tính-kỹ-thuật-32-động-cơ-servo--tỷ-số-truyền)
4. [Mô hình Va chạm, Lưới Hình học (Geometries) & Hệ số Ma sát](#4-mô-hình-va-chạm-lưới-hình-học--hệ-số-ma-sát)
5. [Cấu hình Cảm biến Tích hợp (Sensors & Actuators Pipeline)](#5-cấu-hình-cảm-biến-tích-hợp)
6. [Phân tích Thế Đứng Danh định (Stand Keyframe Kinematics)](#6-phân-tích-thế-đứng-danh-định)

---

## 1. NGUỒN GỐC & TIÊU CHUẨN MÔ HÌNH

Mô hình robot được sử dụng trong dự án được trích xuất từ kho lưu trữ chuẩn hóa quốc tế **MuJoCo Menagerie** của **Google DeepMind**, phát triển dựa trên thiết kế nguyên bản của hãng chế tạo robot **Apptronik** (Austin, Texas, Hoa Kỳ).

### Các Tệp Tin Cấu Thành Chính:
- [`google_deepmind_menagerie/apptronik_apollo/scene.xml`](file:///d:/GitHub/medical-science/google_deepmind_menagerie/apptronik_apollo/scene.xml): Tệp thế giới tổng thể định nghĩa sàn phẳng, ánh sáng định hướng, thông số tiếp xúc đàn hồi và thẻ nạp mô hình con.
- [`google_deepmind_menagerie/apptronik_apollo/apollo.xml`](file:///d:/GitHub/medical-science/google_deepmind_menagerie/apptronik_apollo/apollo.xml): Tệp mô tả định dạng XML cho MuJoCo (MJCF) chứa cấu trúc xương khớp, các mắt xích động học, khối lượng, quán tính và giới hạn cơ khí.
- `assets/`: Thư mục chứa các tệp lưới 3D bề mặt định dạng `.obj` siêu chi tiết và tệp kết cấu vật liệu.

---

## 2. CẤU TRÚC CÂY ĐỘNG HỌC & DANH MỤC 37 PHÂN ĐOẠN THÂN

Robot Apollo gồm **37 phân đoạn vật rắn (bodies)** được liên kết theo cấu trúc hình cây (Kinematic Tree). Dưới đây là bảng trích xuất số liệu khối lượng và vị trí lắp đặt chính xác từ tệp tin mô hình:

| ID | Tên Phân Đoạn Thân (`body_name`) | Khối Lượng ($kg$) | Tọa Độ Gốc Lắp Đặt $[x, y, z]$ ($m$) | Nhóm Giải Phẫu |
| :-: | :--- | :---: | :---: | :--- |
| **00** | `world` | $0.000$ | $[0.000, 0.000, 0.000]$ | Hệ quy chiếu gốc thế giới |
| **01** | `base_link` (Khung chậu / Pelvis) | **$7.436$** | $[0.000, 0.000, 1.081]$ | Khung chậu trung tâm |
| **02** | `torso_oak_d_pro_w_rear_frame_link` | $0.115$ | $[-0.164, 0.000, 0.014]$ | Cảm biến camera sau |
| **03** | `torso_oak_d_pro_w_front_frame_link` | $0.115$ | $[0.087, 0.000, 0.048]$ | Cảm biến camera trước |
| **04** | `torso_roll_link` | $0.824$ | $[0.030, 0.000, 0.031]$ | Khớp lật thân ngang |
| **05** | `torso_pitch_link` | $0.313$ | $[0.000, 0.000, 0.000]$ | Khớp cúi ngửa thân |
| **06** | `torso_link` (Khoang ngực & Pin) | **$19.341$** | $[0.000, 0.000, 0.000]$ | Phân đoạn nặng nhất thân trên |
| **07** | `neck_yaw_link` | $0.708$ | $[-0.030, 0.000, 0.328]$ | Cổ xoay ngang |
| **08** | `neck_roll_link` | $0.019$ | $[-0.025, 0.000, 0.200]$ | Cổ nghiêng bên |
| **09** | `neck_pitch_link` (Cụm đầu & Màn hình) | $1.781$ | $[0.000, 0.000, 0.000]$ | Cụm đầu robot |
| **10** | `l_shoulder_aa_link` | $0.098$ | $[-0.050, 0.200, 0.320]$ | Khớp vai trái dang/khép |
| **11** | `l_shoulder_ie_link` | $0.451$ | $[0.000, 0.000, 0.000]$ | Khớp vai trái xoay trong/ngoài |
| **12** | `l_shoulder_fe_link` (Bắp tay trái) | $3.513$ | $[0.010, 0.039, 0.000]$ | Cánh tay trên trái |
| **13** | `l_elbow_fe_link` (Cẳng tay trái) | $0.948$ | $[0.025, 0.000, -0.315]$ | Khớp khuỷu tay trái |
| **14** | `l_wrist_roll_link` | $0.694$ | $[-0.040, 0.000, -0.060]$ | Cổ tay trái xoay tròn |
| **15** | `l_wrist_yaw_link` | $0.076$ | $[0.000, 0.000, 0.000]$ | Cổ tay trái xoay ngang |
| **16** | `l_wrist_pitch_link` (Bàn tay trái) | $0.686$ | $[0.000, 0.000, 0.000]$ | Cụm bàn tay trái |
| **17** | `r_shoulder_aa_link` | $0.098$ | $[-0.050, -0.200, 0.320]$ | Khớp vai phải dang/khép |
| **18** | `r_shoulder_ie_link` | $0.451$ | $[0.000, 0.000, 0.000]$ | Khớp vai phải xoay trong/ngoài |
| **19** | `r_shoulder_fe_link` (Bắp tay phải) | $3.513$ | $[0.010, -0.039, 0.000]$ | Cánh tay trên phải |
| **20** | `r_elbow_fe_link` (Cẳng tay phải) | $0.948$ | $[0.025, 0.000, -0.315]$ | Khớp khuỷu tay phải |
| **21** | `r_wrist_roll_link` | $0.694$ | $[-0.040, 0.000, -0.060]$ | Cổ tay phải xoay tròn |
| **22** | `r_wrist_yaw_link` | $0.076$ | $[0.000, 0.000, 0.000]$ | Cổ tay phải xoay ngang |
| **23** | `r_wrist_pitch_link` (Bàn tay phải) | $0.686$ | $[0.000, 0.000, 0.000]$ | Cụm bàn tay phải |
| **24** | `l_hip_ie_link` | $0.217$ | $[0.000, 0.125, -0.060]$ | Khớp háng trái xoay |
| **25** | `l_hip_aa_link` | $0.852$ | $[0.000, 0.000, 0.000]$ | Khớp háng trái dang/khép |
| **26** | `l_hip_fe_link` (Đùi trái) | **$8.214$** | $[0.000, 0.055, 0.000]$ | Đùi chi dưới trái |
| **27** | `l_knee_fe_link` (Cẳng chân trái) | **$3.325$** | $[0.000, 0.000, -0.400]$ | Khớp gối và bắp chân trái |
| **28** | `l_ankle_ie_link` | $0.150$ | $[0.000, 0.000, -0.400]$ | Cổ chân trái nghiêng |
| **29** | `l_ankle_pd_link` (Bàn chân trái) | **$2.096$** | $[0.000, 0.000, 0.000]$ | Bàn chân tiếp xúc sàn trái |
| **30** | `r_hip_ie_link` | $0.217$ | $[0.000, -0.125, -0.060]$ | Khớp háng phải xoay |
| **31** | `r_hip_aa_link` | $0.852$ | $[0.000, 0.000, 0.000]$ | Khớp háng phải dang/khép |
| **32** | `r_hip_fe_link` (Đùi phải) | **$8.214$** | $[0.000, -0.055, 0.000]$ | Đùi chi dưới phải |
| **33** | `r_knee_fe_link` (Cẳng chân phải) | **$3.325$** | $[0.000, 0.000, -0.400]$ | Khớp gối và bắp chân phải |
| **34** | `r_ankle_ie_link` | $0.150$ | $[0.000, 0.000, -0.400]$ | Cổ chân phải nghiêng |
| **35** | `r_ankle_pd_link` (Bàn chân phải) | **$2.096$** | $[0.000, 0.000, 0.000]$ | Bàn chân tiếp xúc sàn phải |
| **36** | `stand_link` | $0.000$ | $[0.000, 0.000, 0.000]$ | Khung hỗ trợ giá treo mô phỏng |

---

## 3. ĐẶC TÍNH KỸ THUẬT 32 ĐỘNG CƠ SERVO

Hệ thống truyền động của Apollo sử dụng các động cơ điện mô-men xoắn cao kết hợp bộ giảm tốc trục vít và hộp số hành tinh hiệu suất cao:

```
[ Tín hiệu Điều khiển ] ──> [ Bộ giới hạn dải góc ctrlrange ] ──> [ Bộ khuếch đại mô-men forcerange ]
```

### 3.1. Các Khớp Trọng Tải Cao (High-Torque Joints)
- **Khớp Háng Dang/Khép (`l_hip_aa`, `r_hip_aa`):**  
  - Giới hạn lực: **$\pm 494.0\text{ Nm}$**  
  - Vai trò: Chịu toàn bộ tải trọng lật ngang của cơ thể khi đứng trên một chân hoặc khi chịu lực xô đẩy từ cạnh bên.
- **Khớp Lật Thân Ngang (`torso_roll`):**  
  - Giới hạn lực: **$\pm 414.0\text{ Nm}$**  
  - Vai trò: Điều chỉnh độ thẳng đứng của cột sống so với khung chậu.
- **Khớp Gập Đùi (`l_hip_fe`, `r_hip_fe`):**  
  - Giới hạn lực: **$\pm 342.0\text{ Nm}$**  
  - Dải góc: $[-1.85, 0.48]\text{ rad}$ (tương đương $[-106^\circ, +27.5^\circ]$).
- **Khớp Gối (`l_knee_fe`, `r_knee_fe`):**  
  - Giới hạn lực: **$\pm 336.0\text{ Nm}$**  
  - Dải góc: $[0.00, 2.62]\text{ rad}$ (tương đương $[0^\circ, 150^\circ]$ — gập một chiều).

### 3.2. Bảng Thông Số Chi Tiết Toàn Bộ 32 Actuator:

| Nhóm Cơ Thể | Tên Actuator | Dải Góc Điều Khiển $[rad]$ | Giới Hạn Mô-men $[Nm]$ |
| :--- | :--- | :---: | :---: |
| **Thân (Torso)** | `torso_yaw` | $[-0.83, 0.83]$ | $\pm 120.0$ |
| | `torso_roll` | $[-0.21, 0.21]$ | $\pm 414.0$ |
| | `torso_pitch` | $[-0.31, 1.35]$ | $\pm 315.0$ |
| **Cổ (Neck)** | `neck_yaw` | $[-1.66, 1.66]$ | $\pm 10.6$ |
| | `neck_roll` | $[-0.79, 0.79]$ | $\pm 34.2$ |
| | `neck_pitch` | $[-0.26, 0.52]$ | $\pm 34.2$ |
| **Chi Trên Trái** | `l_shoulder_aa` | $[-0.12, 1.61]$ | $\pm 78.0$ |
| | `l_shoulder_ie` | $[-0.47, 0.47]$ | $\pm 67.0$ |
| | `l_shoulder_fe` | $[-2.18, 0.61]$ | $\pm 114.0$ |
| | `l_elbow_fe` | $[-2.62, 0.17]$ | $\pm 114.0$ |
| | `l_wrist_roll` | $[-1.66, 1.66]$ | $\pm 10.6$ |
| | `l_wrist_yaw` | $[-0.79, 0.79]$ | $\pm 34.2$ |
| | `l_wrist_pitch` | $[-0.84, 1.68]$ | $\pm 34.2$ |
| **Chi Trên Phải** | `r_shoulder_aa` | $[-1.61, 0.12]$ | $\pm 78.0$ |
| | `r_shoulder_ie` | $[-0.47, 0.47]$ | $\pm 67.0$ |
| | `r_shoulder_fe` | $[-2.18, 0.61]$ | $\pm 114.0$ |
| | `r_elbow_fe` | $[-2.62, 0.17]$ | $\pm 114.0$ |
| | `r_wrist_roll` | $[-1.66, 1.66]$ | $\pm 10.6$ |
| | `r_wrist_yaw` | $[-0.79, 0.79]$ | $\pm 34.2$ |
| | `r_wrist_pitch` | $[-1.68, 0.84]$ | $\pm 34.2$ |
| **Chi Dưới Trái** | `l_hip_ie` | $[-0.57, 1.09]$ | $\pm 120.0$ |
| | `l_hip_aa` | $[-0.22, 0.74]$ | $\pm 494.0$ |
| | `l_hip_fe` | $[-1.85, 0.48]$ | $\pm 342.0$ |
| | `l_knee_fe` | $[ 0.00, 2.62]$ | $\pm 336.0$ |
| | `l_ankle_ie` | $[-0.65, 0.31]$ | $\pm 120.0$ |
| | `l_ankle_pd` | $[-1.57, 0.44]$ | $\pm 150.0$ |
| **Chi Dưới Phải** | `r_hip_ie` | $[-1.09, 0.57]$ | $\pm 120.0$ |
| | `r_hip_aa` | $[-0.74, 0.22]$ | $\pm 494.0$ |
| | `r_hip_fe` | $[-1.85, 0.48]$ | $\pm 342.0$ |
| | `r_knee_fe` | $[ 0.00, 2.62]$ | $\pm 336.0$ |
| | `r_ankle_ie` | $[-0.31, 0.65]$ | $\pm 120.0$ |
| | `r_ankle_pd` | $[-1.57, 0.44]$ | $\pm 150.0$ |

---

## 4. MÔ HÌNH VA CHẠM & HỆ SỐ MA SÁT TIẾP XÚC

Mô hình gồm **80 hình học (geometries)**, chia thành 2 nhóm:
1. **Visual Geoms (`group=2`):** Mắt lưới độ phân giải cao phục vụ kết xuất đồ họa quang học. Không tham gia tính toán va chạm để tiết kiệm tài nguyên GPU.
2. **Collision Geoms (`group=3`):** Bao gồm các khối hình học cơ bản lồi (Convex Primitives: Cylinders, Boxes, Spheres, Capsules) được gắn thẻ `contype="1" conaffinity="1"`.

### Cấu Hình Tương Tác Mặt Đất Chuẩn Trong `scene.xml`:
```xml
<default>
    <geom friction="0.8 0.005 0.0001" solref="0.004 1" solimp="0.9 0.95 0.001 0.5 2"/>
</default>
```
- **Hệ số ma sát trượt ($\mu_{sliding} = 0.8$):** Tương đương đế cao su tiếp xúc với mặt sàn gỗ hoặc bê tông nhẵn.
- **Hệ số ma sát xoắn ($\mu_{torsional} = 0.005$):** Ngăn cản hiện tượng xoay trượt tự do quanh trục thẳng đứng.
- **Tham số đàn hồi (`solref = [0.004, 1.0]`):** Thời gian phục hồi đàn hồi cực ngắn (4 ms) và hệ số cản tới hạn (Damping ratio = 1.0) đảm bảo bàn chân không bị nảy tưng tưng khi chạm đất.

---

## 5. CẤU HÌNH CẢM BIẾN TÍCH HỢP

Robot được trang bị hệ thống cảm biến mô phỏng hoàn chỉnh tương đương phần cứng thật:

1. **Cảm biến Quán tính IMU Thân (`imu_pelvis`):**
   - Đặt ngay tại trọng tâm khung chậu (`base_link`).
   - Cung cấp: Gia tốc tuyến tính 3 chiều $[a_x, a_y, a_z]$ và vận tốc góc quay 3 chiều $[\omega_x, \omega_y, \omega_z]$.
2. **Cảm biến Vị trí và Vận tốc Khớp (Encoders):**
   - Đọc trực tiếp góc quay $q_i$ và vận tốc góc $\dot{q}_i$ cho toàn bộ 32 khớp với tần số 500 Hz.
3. **Cảm biến Tiếp xúc Bàn chân (Foot Sole Contact Sensors):**
   - Xác định tổng lực tiếp xúc pháp tuyến $F_z$ và vị trí tâm áp lực (CoP) trên mặt phẳng đế.

---

## 6. PHÂN TÍCH THẾ ĐỨNG DANH ĐỊNH (STAND KEYFRAME)

Trong tệp tin [`scene.xml`](file:///d:/GitHub/medical-science/google_deepmind_menagerie/apptronik_apollo/scene.xml), trạng thái `keyframe` mang tên **"stand"** xác định cấu hình thăng bằng tĩnh tối ưu của robot:

- **Tọa độ vị trí gốc:** $[x_0, y_0, z_0] = [0.0, 0.0, 1.0160\text{ m}]$.
- **Hướng Quaternion gốc:** $[q_w, q_x, q_y, q_z] = [1.0, 0.0, 0.0, 0.0]$ (hoàn toàn thẳng đứng, không nghiêng).
- **Góc chùng gối sinh học:** Hai đầu gối hơi chùng một góc nhỏ $\approx 0.15\text{ rad}$ ($\approx 8.6^\circ$) để tránh hiện tượng kỳ dị động học (Singularity) khi chân duỗi thẳng đơ, giúp hệ thống sẵn sàng co duỗi cơ bắp phản xạ khi chịu ngoại lực đẩy xô.
