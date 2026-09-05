# 📐 TÀI LIỆU KỸ THUẬT 01: LÝ THUYẾT CƠ SINH HỌC & HỆ THỐNG VIỄN TRẮC (BIOMECHANICS & TELEMETRY)

> **Dự án**: Apptronik Apollo Humanoid Robotics (`medical-science`)  
> **Chuyên đề**: Động học, Động lực học Tiếp xúc, Cân bằng Toàn thân & Hệ thống Viễn trắc Đồ họa Thời gian thực  
> **Mã tài liệu**: `DOCX-BIO-01` | **Phiên bản**: 2.4.0

---

## 📑 MỤC LỤC
1. [Tổng quan Cơ sinh học Robot Hình nhân (Humanoid Biomechanics)](#1-tổng-quan-cơ-sinh-học-robot-hình-nhân)
2. [Cơ sở Toán học Trọng tâm Toàn thân (Center of Mass - CoM)](#2-cơ-sở-toán-học-trọng-tâm-toàn-thân-com)
3. [Lý thuyết & Công thức Điểm Không Mô-men (Zero Moment Point - ZMP)](#3-lý-thuyết--công-thức-điểm-không-mô-men-zmp)
4. [Lực Phản lực Mặt đất (Ground Reaction Force - GRF) & Nón Ma sát](#4-lực-phản-lực-mặt-đất-grf--nón-ma-sát)
5. [Đa giác Thăng bằng (Support Polygon) & Biên Độ Ổn định Động](#5-đa-giác-thăng-bằng-support-polygon--biên-độ-ổn-định-động)
6. [Hệ thống Đo lường Viễn trắc Thời gian thực trong `main.py`](#6-hệ-thống-đo-lường-viễn-trắc-thời-gian-thực-trong-mainpy)
7. [Kiến trúc Hiển thị Đồ họa 3D OpenGL & Dao động ký 4 Kênh](#7-kiến-trúc-hiển-thị-đồ-họa-3d-opengl--dao-động-ký-4-kênh)

---

## 1. TỔNG QUAN CƠ SINH HỌC ROBOT HÌNH NHÂN

Robot hình người (Humanoid Robot) như **Apptronik Apollo** là một hệ thống đa vật thể liên kết hở (Open-chain multi-body mechanical system) có gốc tự do (Free-floating base) di chuyển trong không gian 3 chiều. 

Khác với các cánh tay robot công nghiệp có đế cố định xuống mặt sàn (Fixed-base manipulators), robot hình nhân chỉ có thể tương tác với môi trường thông qua **tiếp xúc bàn chân không liên tục (Unilateral Contact Constraints)**:
- Mặt đất chỉ có thể **đẩy** bàn chân (lực pháp tuyến $F_z \ge 0$), không thể kéo hoặc giữ dính bàn chân xuống đất.
- Lực ma sát tiếp xúc tiếp tuyến bị giới hạn bởi nón ma sát Coulomb: $\|F_{xy}\| \le \mu F_z$.

Do đó, mục tiêu cốt lõi của bài toán điều khiển thăng bằng là đảm bảo các mô-men quán tính và gia tốc trọng trường không làm lật bàn chân quanh các cạnh của đa giác tiếp xúc.

```
       [ Khung Thân & Đầu: ~30.2 kg ]
                   │
         [ Khung Chậu: 7.436 kg ] ──> Cảm biến IMU (Góc Roll / Pitch / Yaw)
              ┌────┴────┐
              ▼         ▼
     [ Chi Trái ]     [ Chi Phải ]   ──> 6 DoF mỗi chân (Háng: 3, Gối: 1, Cổ chân: 2)
              │         │
              ▼         ▼
       [ Bàn Chân ]   [ Bàn Chân ]   ──> Tiếp xúc mặt sàn (GRF: Fz_left + Fz_right ≈ M * g)
```

---

## 2. CƠ SỞ TOÁN HỌC TRỌNG TÂM TOÀN THÂN (COM)

Trọng tâm toàn thân (Center of Mass - CoM) là điểm hình học trọng yếu đại diện cho sự phân bố khối lượng của toàn bộ 37 phân đoạn cơ thể robot trong không gian thế giới.

### 2.1. Công thức Tọa độ Trọng tâm 3D
Với một hệ gồm $N = 37$ vật rắn (bodies), mỗi vật có khối lượng $m_i$ và vị trí trọng tâm cục bộ trong hệ quy chiếu thế giới là $\mathbf{x}_i = [x_i, y_i, z_i]^T$:

$$\mathbf{r}_{CoM} = \frac{1}{M} \sum_{i=1}^{N} m_i \mathbf{x}_i \quad \text{với} \quad M = \sum_{i=1}^{N} m_i = 80.898\text{ kg}$$

Trong mã nguồn [`main.py`](file:///d:/GitHub/medical-science/main.py), tọa độ này được trích xuất trực tiếp từ bộ đệm của MuJoCo:
```python
# Trích xuất vị trí trọng tâm từ MuJoCo C-Data:
com_pos = np.sum(data.xipos * model.body_mass[:, None], axis=0) / np.sum(model.body_mass)
```

### 2.2. Vận tốc và Gia tốc Trọng tâm
Đạo hàm bậc một và bậc hai của véc-tơ CoM theo thời gian mô tả động lực học tịnh tiến toàn thân:

$$\dot{\mathbf{r}}_{CoM} = \mathbf{v}_{CoM} = \frac{1}{M} \sum_{i=1}^{N} m_i \dot{\mathbf{x}}_i$$

$$\ddot{\mathbf{r}}_{CoM} = \mathbf{a}_{CoM} = \frac{1}{M} \sum_{i=1}^{N} m_i \ddot{\mathbf{x}}_i$$

Theo định luật II Newton cho hệ đa vật thể, tổng ngoại lực tác dụng lên robot (bao gồm trọng lực và lực phản lực mặt đất) liên hệ trực tiếp với gia tốc trọng tâm:

$$M (\ddot{\mathbf{r}}_{CoM} + \mathbf{g}) = \sum_{k \in \mathcal{C}} \mathbf{f}_k^{GRF}$$

Trong đó $\mathbf{g} = [0, 0, 9.81]^T\text{ m/s}^2$ và $\mathbf{f}_k^{GRF}$ là các véc-tơ phản lực tiếp xúc tại điểm tiếp xúc $k$.

---

## 3. LÝ THUYẾT & CÔNG THỨC ĐIỂM KHÔNG MÔ-MEN (ZMP)

Điểm Zero Moment Point (ZMP), do giáo sư Miomir Vukobratović đề xuất năm 1968, là tiêu chuẩn vàng định lượng sự ổn định động lực học của robot hai chân.

### 3.1. Định nghĩa Vật lý
ZMP là điểm nằm trên mặt phẳng tiếp xúc (thường là mặt sàn $z=0$), tại đó **tổng mô-men của các lực quán tính và trọng trường tác dụng lên robot hoàn toàn triệt tiêu theo hai trục nằm ngang $X$ và $Y$**:

$$\tau_x^{net}(P_{ZMP}) = 0, \quad \tau_y^{net}(P_{ZMP}) = 0$$

### 3.2. Công thức Động Lực Học Rút Gọn
Khi robot di chuyển hoặc giữ thăng bằng trên mặt phẳng ngang, tọa độ ZMP $[x_{zmp}, y_{zmp}, 0]^T$ được tính bằng:

$$x_{zmp} = \frac{\sum_{i=1}^{N} m_i (\ddot{z}_i + g) x_i - \sum_{i=1}^{N} m_i \ddot{x}_i z_i - \sum_{i=1}^{N} I_{yy, i} \dot{\omega}_{y, i}}{\sum_{i=1}^{N} m_i (\ddot{z}_i + g)}$$

$$y_{zmp} = \frac{\sum_{i=1}^{N} m_i (\ddot{z}_i + g) y_i - \sum_{i=1}^{N} m_i \ddot{y}_i z_i - \sum_{i=1}^{N} I_{xx, i} \dot{\omega}_{x, i}}{\sum_{i=1}^{N} m_i (\ddot{z}_i + g)}$$

Trong đó:
- $m_i$: Khối lượng phân đoạn thân thứ $i$.
- $[x_i, y_i, z_i]^T$: Vị trí tức thời của tâm khối lượng phân đoạn thứ $i$.
- $[\ddot{x}_i, \ddot{y}_i, \ddot{z}_i]^T$: Gia tốc tịnh tiến tức thời của phân đoạn thứ $i$.
- $I_{xx, i}, I_{yy, i}$: Mô-men quán tính khối lượng quanh các trục chính.
- $\dot{\omega}_i$: Gia tốc góc quay của phân đoạn.

### 3.3. Mô hình Con lắc Ngược Tuyến tính (LIPM Approximation)
Khi xấp xỉ chuyển động của robot thành một con lắc ngược với độ cao trọng tâm không đổi ($z_{CoM} = z_c = \text{const} \approx 1.016\text{ m}$), công thức ZMP đơn giản hóa thành:

$$x_{zmp} \approx x_{CoM} - \frac{z_c}{g} \ddot{x}_{CoM}$$

$$y_{zmp} \approx y_{CoM} - \frac{z_c}{g} \ddot{y}_{CoM}$$

Biểu thức này chỉ ra rằng: **Để giữ ZMP ở trung tâm bàn chân, bất cứ khi nào thân trên có gia tốc về phía trước ($\ddot{x} > 0$), ZMP sẽ bị dịch chuyển về phía sau so với CoM.**

---

## 4. LỰC PHẢN LỰC MẶT ĐẤT (GRF) & NÓN MA SÁT

### 4.1. Phân bố Lực Tiếp Xúc
Bàn chân của Apollo có kích thước chiều dài $L = 0.28\text{ m}$ và chiều rộng $W = 0.14\text{ m}$. Khi đứng bằng hai chân (Double Support Phase), tải trọng lượng $M \cdot g = 80.898 \times 9.80665 \approx 793.33\text{ N}$ được phân chia đều cho 2 chân ở trạng thái tĩnh:

$$F_{z, left} \approx 396.6\text{ N}, \quad F_{z, right} \approx 396.6\text{ N}$$

Trong [`main.py`](file:///d:/GitHub/medical-science/main.py), lực pháp tuyến $F_z$ được tính bằng cách tích hợp toàn bộ các điểm tiếp xúc chủ động thuộc bàn chân trái (`l_foot`) và bàn chân phải (`r_foot`):

```python
fz_left, fz_right = 0.0, 0.0
for i in range(data.ncon):
    contact = data.contact[i]
    # Xác định geom tiếp xúc thuộc chi dưới nào
    force_vec = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force_vec)
    normal_force = force_vec[0]  # Lực dọc pháp tuyến tiếp xúc
    if is_left_foot(contact.geom1) or is_left_foot(contact.geom2):
        fz_left += normal_force
    elif is_right_foot(contact.geom1) or is_right_foot(contact.geom2):
        fz_right += normal_force
```

### 4.2. Ràng Buộc Nón Ma Sát (Coulomb Friction Cone)
Để bàn chân không bị trượt trên sàn, tỷ số giữa lực ma sát ngang và lực nén vuông góc phải nằm trong giới hạn ma sát tĩnh $\mu = 0.8$:

$$\sqrt{F_x^2 + F_y^2} \le \mu F_z$$

Trong bộ giải tiếp xúc MuJoCo (`geom_solref` = `[0.004, 1.0]`, `geom_solimp` = `[0.9, 0.95, 0.001, 0.5, 2.0]`), các lực trượt vi mô được xấp xỉ qua ma trận nón ma sát hình chóp (Pyramidal Cone Approximation) để tối ưu hóa thời gian tính toán trên GPU.

---

## 5. ĐA GIÁC THĂNG BẰNG & BIÊN ĐỘ ỔN ĐỊNH ĐỘNG

### 5.1. Định nghĩa Đa giác Thăng bằng (Support Polygon - BoS)
Đa giác thăng bằng là bao lồi (Convex Hull) của tất cả các điểm tiếp xúc vật lý giữa bàn chân robot và mặt sàn:
- **Thế đứng 2 chân (Double Support):** Bao lồi gồm toàn bộ diện tích hai bàn chân và vùng sàn nằm giữa hai bàn chân.
- **Thế đứng 1 chân (Single Support):** Bao lồi thu hẹp lại chỉ bằng chu vi bàn chân đang chống đất.

```
       CHÂN TRÁI                           CHÂN PHẢI
   ┌───────────────┐                   ┌───────────────┐
   │ x1, y1        │                   │ x3, y3        │
   │               │ <─── VÙNG ĐA GIÁC ───>            │
   │       ZMP     │      THĂNG BẰNG   │               │
   │ x2, y2        │     (CONVEX HULL) │ x4, y4        │
   └───────────────┘                   └───────────────┘
```

### 5.2. Tiêu Chuẩn Thăng Bằng Vukobratovic
- **Robot Ổn Định Tuyệt Đối:** $P_{ZMP} \in \text{Interior}(\mathcal{P}_{BoS})$ (Điểm ZMP nằm nghiêm ngặt bên trong đa giác thăng bằng).
- **Nguy Cơ Lật Đổ Cơ Học (Marginal Stability):** $P_{ZMP} \in \partial \mathcal{P}_{BoS}$ (Điểm ZMP chạm vào mép biên của đa giác thăng bằng $\implies$ Bàn chân bắt đầu xoay quanh mép ngoài).
- **Mất Thăng Bằng & Ngã (Instability):** Không tồn tại nghiệm vật lý thỏa mãn ZMP bên trong $\mathcal{P}_{BoS}$ $\implies$ Xuất hiện mô-men lật tự do (Tipping Moment) quanh cạnh bàn chân.

Biên độ ổn định động (Dynamic Stability Margin) được đo bằng khoảng cách Euclid nhỏ nhất từ điểm ZMP đến mép biên đa giác:

$$d_{margin} = \min_{e \in \partial \mathcal{P}_{BoS}} \text{distance}(P_{ZMP}, e)$$

---

## 6. HỆ THỐNG ĐO LƯỜNG VIỄN TRẮC THỜI GIAN THỰC TRONG `MAIN.PY`

Lớp [`BiomechanicsTelemetry`](file:///d:/GitHub/medical-science/main.py) trong file `main.py` thực hiện chu trình cập nhật 100 lần/giây (100 Hz), thu thập và tính toán toàn bộ các biến số trạng thái:

```python
class BiomechanicsTelemetry:
    def update(self, data):
        # 1. Trọng tâm và vận tốc CoM
        com = np.sum(data.xipos * self.masses[:, None], axis=0) / self.total_mass
        com_vel = np.sum(data.cvel[:, 3:6] * self.masses[:, None], axis=0) / self.total_mass
        
        # 2. Véc-tơ đơn vị thân (Upvector) từ Quaternion
        qw, qx, qy, qz = data.qpos[3:7]
        upvec = np.array([
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx**2 + qy**2)
        ])
        
        # 3. Phản lực bàn chân GRF
        fz_l, fz_r = self._compute_foot_forces(data)
        
        # 4. Điểm ZMP
        zmp_x, zmp_y = self._compute_zmp(data, com, com_vel)
        
        return {
            'com_pos': com, 'com_vel': com_vel,
            'upvector': upvec, 'pelvis_z': data.qpos[2],
            'fz_left': fz_l, 'fz_right': fz_r,
            'zmp': np.array([zmp_x, zmp_y, 0.0])
        }
```

---

## 7. KIẾN TRÚC HIỂN THỊ ĐỒ HỌA 3D OPENGL & DAO ĐỘNG KÝ 4 KÊNH

Giao diện người dùng trong `main.py` được xây dựng bằng pipeline kết xuất đồ họa kép:

### 7.1. Kết xuất 3D Phối Cảnh (3D Perspective Pass)
- Sử dụng hàm `mujoco.mjr_render()` để vẽ khung cảnh 3D của robot Apollo với ánh sáng định hướng, bóng đổ mặt sàn tự nhiên.
- Toàn bộ các hình học nhân tạo trang trí (hình cầu CoM giả ở háng, vệt ZMP rải sàn) đã được loại bỏ triệt để nhằm giữ nguyên tính trực quan cơ khí nguyên bản.

### 7.2. Kết xuất 2D Trực Giao (2D Orthographic HUD Pass)
- Chuyển đổi ma trận chiếu sang tọa độ màn hình trực giao: `glOrtho(0, width, height, 0, -1, 1)`.
- **Hệ thống Font Chữ Tiếng Việt Unicode:** Sử dụng lớp `FontRenderer` đọc font `segoeuib.ttf` của Windows, kết hợp thư viện Pillow tạo Texture động hỗ trợ đầy đủ các ký tự tiếng Việt có dấu (`ắ, ằ, ộ, ệ, ư, ơ`) chống răng cưa siêu nét.
- **Dao Động Ký 4 Kênh (Real-Time Oscilloscope):**
  - Kênh 1 (Xanh lam): Độ cao khung chậu Pelvis Z (danh định 1.016 m).
  - Kênh 2 (Xanh lục): Lực tiếp xúc chân trái $F_{z, left}$ (danh định ~396 N).
  - Kênh 3 (Cam đỏ): Lực tiếp xúc chân phải $F_{z, right}$ (danh định ~396 N).
  - Kênh 4 (Vàng nhạt): Vận tốc trượt ngang $V_y$ CoM (mục tiêu triệt tiêu về 0.0 m/s).
- **Quả Cầu Định Hướng Gizmo 3D (Blender-Style):** Luôn hiển thị cố định ở góc trên bên phải màn hình, đồng bộ góc xoay tự do với camera MuJoCo thông qua phép nhân ma trận quay thế giới $[R, U, F]$.
