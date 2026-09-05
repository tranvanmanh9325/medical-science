# 🤖 Apptronik Apollo Humanoid — Biomechanics Telemetry & GPU Reinforcement Learning Suite

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.12.0%2B-orange?logo=google&logoColor=white)](https://mujoco.org/)
[![JAX / MJX](https://img.shields.io/badge/JAX%20%2F%20MJX-GPU%20Accelerated-crimson?logo=google&logoColor=white)](https://github.com/google-deepmind/mujoco)
[![Flax / Optax](https://img.shields.io/badge/Flax-PPO%20Actor--Critic-blueviolet)](https://github.com/google/flax)
[![Kaggle Dual T4](https://img.shields.io/badge/Kaggle-Dual%20NVIDIA%20T4-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU%20Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-green)](LICENSE)

---

## 📌 1. Tổng Quan Dự Án (Project Overview)

**`medical-science`** là nền tảng nghiên cứu chuyên sâu kết hợp giữa **Cơ sinh học Robot Hình người (Humanoid Biomechanics)**, **Điều khiển Toàn thân (Whole-Body Locomotion)** và **Học tăng cường gia tốc phần cứng trên GPU (GPU-Accelerated Reinforcement Learning)**. 

Dự án sử dụng mô hình robot hình người công nghiệp thế hệ mới **Apptronik Apollo** (32 bậc tự do - DoF, cao 1.73 m, khối lượng 73 kg) từ bộ thư viện chuẩn của **Google DeepMind Menagerie**, đồng thời tích hợp các tài nguyên nghiên cứu robot phẫu thuật y sinh (**da Vinci Research Kit / dVRK**).

### 🎯 Mục Tiêu Nghiên Cứu
1. **Khám phá Cơ sinh học & Đo lường Động học thời gian thực:** Xây dựng phòng thí nghiệm mô phỏng 3D tương tác với độ chính xác vật lý cao, đo lường liên tục Trọng tâm (CoM), Lực phản lực mặt đất (GRF), Điểm ZMP và đa giác thăng bằng.
2. **Huấn luyện Bộ não Điều khiển Cân bằng Đứng thẳng (Stage 1 Standing Balance):** Áp dụng thuật toán PPO (Proximal Policy Optimization) chạy song song trên **4.096 môi trường MuJoCo MJX** bằng JAX, đạt tốc độ hơn **500.000 bước/giây** trên GPU đám mây (Kaggle & Google Colab).
3. **Khả năng Chống Nhiễu loạn Ngoại lực (Disturbance Rejection):** Tự động khôi phục tư thế cân bằng khi bị xô đẩy mạnh mà không bị sụp đổ cơ học.

---

## 🏛️ 2. Kiến Trúc Kỹ Thuật Hai Tầng (Hierarchical Architecture)

Hệ thống điều khiển được thiết kế theo cấu trúc sinh học phân tầng tương tự hệ thần kinh người:

```
                          [ LỆNH MỤC TIÊU CẤP CAO ]
                    "Đi tới bàn khám, né chướng ngại vật"
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. ĐẠI NÃO TƯ DUY & NHẬN THỨC (High-Level Cognitive Brain — System 2)      │
│  - Mô hình Đa phương thức (VLA / LLM: Gemini 1.5 Pro, Claude, DeepSeek)     │
│  - Tần số: 1 - 2 Hz (0.5s - 1.0s / chu kỳ)                                  │
│  - Nhiệm vụ: Xử lý giọng nói, camera thị giác 3D, lập kế hoạch hành động    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Gửi lệnh vận tốc (vx, vy, yaw_rate)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. TIỂU NÃO VẬN ĐỘNG & THĂNG BẰNG (Low-Level Motor Brain — System 1)       │
│  - Mạng nơ-ron PPO Actor-Critic (Flax / JAX — Huấn luyện bằng MuJoCo MJX)   │
│  - Tần số: 100 - 500 Hz (0.002s - 0.01s / chu kỳ)                           │
│  - Nhiệm vụ: Giữ vững trọng tâm (CoM), điều khiển mô-men xoắn 32 khớp,      │
│              phản xạ chống xô ngã, duy trì tư thế thẳng đứng.               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ 3. Phòng Thí Nghiệm Mô Phỏng 3D (`main.py`)

Giao diện mô phỏng được phát triển trực tiếp trên nền tảng **OpenGL / GLFW** kết hợp nhân vật lý **MuJoCo C-API**, được Việt hóa 100% các thuật ngữ cơ sinh học với đồ họa khoa học tối giản, trung thực:

### ✨ Các Tính Năng Nổi Bật:
- **Đo lường Cơ sinh học Thời gian thực:**
  - **Trọng tâm toàn thân (Center of Mass - CoM):** Vị trí 3D $(x, y, z)$ và vận tốc dịch chuyển tức thời.
  - **Lực phản lực mặt đất (Ground Reaction Force - GRF):** Véc-tơ lực ép $F_z$ độc lập trên từng bàn chân trái/phải.
  - **Điểm Zero Moment Point (ZMP):** Xác định tâm áp lực tiếp xúc so với đa giác thăng bằng hỗ trợ (Support Polygon).
  - **Cảm biến Góc nghiêng IMU:** Đo lường độ lệch trục Roll, Pitch, Yaw của khung chậu và thân trên.
- **Đồ thị Sóng Dao động 4 Kênh (Real-time Oscilloscope):**
  - Kênh 1: Độ cao khung hông Pelvis Z $(m)$.
  - Kênh 2: Lực ép bàn chân trái $F_z$ Left $(N)$.
  - Kênh 3: Lực ép bàn chân phải $F_z$ Right $(N)$.
  - Kênh 4: Vận tốc trượt ngang $V_y$ CoM $(m/s)$.
- **Bảng Chẩn Đoán Tải Lực 32 Khớp:** Hiển thị lực mô-men xoắn $(Nm)$ chi tiết của các nhóm cơ chính (Háng gập/duỗi, Gối, Cổ chân, Lưng nghiêng/gập, Khớp vai).
- **Quả Cầu Định Hướng 3D Gizmo (Blender-Style):** Cố định ở góc trên bên phải màn hình, hỗ trợ định hướng các trục $+X/-X, +Y/-Y, +Z/-Z$ trong không gian 3D.
- **Không Gian Vật Lý Thuần Khiết:** Đã loại bỏ các khối hình học trang trí nhân tạo, giữ nguyên bóng đổ đổ bóng động (dynamic floor shadow) và hình thể chuẩn của robot.
- **Bảo Vệ Tài Nguyên Phần Cứng:** Tích hợp **Windows Named Mutex** (`Apollo_MuJoCo_Simulation_Mutex`) chống khởi chạy trùng lặp gây quá nhiệt GPU.

### 🎮 Bảng Phím Tắt Điều Khiển:

| Phím | Chức năng chi tiết |
| :---: | :--- |
| **`TAB`** | **Phím duy nhất:** Bật / Tắt đồng thời toàn bộ bảng thông số HUD & đồ thị sóng dao động 2D |
| **`Mũi tên` / `F`** | Tác dụng xung lực đẩy xô thử nghiệm (Push Perturbation) theo các hướng $X/Y$ |
| **`Space`** | Tạm dừng / Tiếp tục mô phỏng vật lý |
| **`R`** | Đặt lại trạng thái ban đầu của Robot |
| **`F8`** | Chuyển đổi giao diện Sáng (Academic Light) / Tối (Dark Studio) |
| **`P`** | Chụp ảnh màn hình độ phân giải cao lưu vào thư mục `pic/` |
| **`ESC` / `Q`** | Thoát ứng dụng an toàn và giải phóng 100% bộ nhớ GPU |

---

## ⚡ 4. Quy Trình Huấn Luyện RL Trên GPU Đám Mây (`training/`)

Quy trình huấn luyện PPO sử dụng công nghệ **MuJoCo MJX** (bản chuyển dịch toán tử MuJoCo sang XLA của Google DeepMind) cho phép vector hóa hàng ngàn môi trường mô phỏng trực tiếp trên bộ nhớ VRAM của GPU:

```
[ 4.096 Môi trường MuJoCo MJX chạy song song trên GPU ]
                          │
                          ▼ (Rollout 32 bước)
[ Mạng Actor-Critic PPO (Flax / JAX) cập nhật Gradients qua Optax ]
                          │
                          ▼
[ Hàm Thưởng CoM + Giữ Hướng Đứng Thẳng + Phạt Tiêu Thụ Năng Lượng ]
                          │
                          ▼
[ Tự động lưu Checkpoint .npz vào checkpoints/ ]
```

### 🏆 Công Thức Hàm Thưởng Cân Bằng (Reward Formulation):
$$\mathcal{R} = \Delta t \cdot \left[ 1.0 \cdot e^{-\frac{\|v_{xy}\|^2}{\sigma}} + 0.5 \cdot e^{-\frac{\omega_z^2}{\sigma}} - 2.0 \cdot v_z^2 - 0.05 \cdot \|\omega_{xy}\|^2 - 1.0 \cdot \|u_{xy}\|^2 - 0.5 \cdot \|q - q_{nom}\|_1 - 10^{-4} \cdot \|\tau\|_2 - 0.01 \cdot \|\Delta a\|^2 - 10.0 \cdot C_{limit} \right]$$

- Duy trì vận tốc ngang tiệm cận 0 và thân hướng thẳng đứng tuyệt đối ($u_z \approx 1$).
- Triệt tiêu dao động độ cao trục $Z$ và hạn chế tiêu hao mô-men xoắn $\tau$.
- Phạt vi phạm biên độ góc giới hạn của các khớp cơ khí ($C_{limit}$).

### ☁️ Triển Khai Đám Mây Kép (Dual-Cloud Workflow):

Dự án hỗ trợ chuyển đổi linh hoạt $1-1$ giữa hai nền tảng GPU Cloud mạnh mẽ nhất:

1. **Kaggle Dual T4 GPU:**
   - Đẩy và chạy ngầm trực tiếp từ dòng lệnh bằng Kaggle API:
     ```powershell
     python training/push_to_kaggle.py
     ```
2. **Google Colab GPU Suite:**
   - Đồng bộ mã nguồn lên GitHub và chạy trên Colab T4:
     ```powershell
     python training/push_to_colab.py
     ```
   - 👉 **[Mở Sổ Tay Huấn Luyện Apollo Trên Google Colab](https://colab.research.google.com/github/tranvanmanh9325/medical-science/blob/main/colab_apollo_training.ipynb)**
   - Cơ chế **Auto-Resume**: Tự động nhận diện file `.npz` cũ để tiếp tục huấn luyện nối tiếp mà không cần chạy lại từ đầu.

---

## 📂 5. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
medical-science/
├── assets/                               # Tài nguyên đồ họa (Textures, Font tiếng Việt Segoe UI)
├── google_deepmind_menagerie/
│   └── apptronik_apollo/                # Mô hình 3D URDF/MJCF chính thức của Robot Apollo
│       ├── scene.xml                     # Sân vận động & cấu hình vật lý thế giới
│       ├── apollo.xml                    # Cấu trúc kinematic 32 khớp, cảm biến & actuator
│       └── assets/                       # File lưới 3D (.obj, .stl) và vật liệu
├── training/                             # Bộ công cụ huấn luyện Reinforcement Learning
│   ├── env_apollo_mjx.py                 # Môi trường vector hóa MJX cho Apollo
│   ├── rewards.py                        # Hàm phần thưởng cơ sinh học thăng bằng
│   ├── ppo_mjx_trainer.py                # Thuật toán PPO Actor-Critic viết bằng Flax & JAX
│   ├── kaggle_train.py                   # Script thực thi trên Kaggle Multi-GPU
│   ├── push_to_kaggle.py                 # Đóng gói và đẩy tự động lên Kaggle CLI
│   ├── colab_train.py                    # Script thực thi trên Google Colab GPU
│   ├── push_to_colab.py                  # Tự động hóa đồng bộ Google Colab
│   └── test_mini_train_sample.py         # Mẫu kiểm thử nhanh (Smoke Test) cục bộ
├── kaggle_kernel_deploy/                 # Gói triển khai hoàn chỉnh cho Kaggle Notebook
│   └── apollo_humanoid_mjx_training.ipynb
├── colab_deploy/                         # Gói triển khai hoàn chỉnh cho Google Colab
│   └── colab_apollo_humanoid_mjx_training.ipynb
├── colab_apollo_training.ipynb           # Notebook 1-Click mở trực tiếp trên Colab
├── davinci_dvrk/                         # Mô hình & tài liệu robot phẫu thuật da Vinci (dVRK)
├── main.py                               # Ứng dụng mô phỏng 3D tương tác & đo lường thời gian thực
├── run.bat                               # Trình khởi chạy 1-Click tự động dọn dẹp tiến trình cho Windows
├── run.ps1                               # Script khởi chạy PowerShell
├── requirements-train.txt                # Danh sách thư viện phụ thuộc Python
├── Dockerfile                            # Môi trường container hóa CUDA 12.2
└── docker-compose.yml                    # Cấu hình Docker Compose đa nền tảng
```

---

## 🚀 6. Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### Yêu Cầu Hệ Thống:
- **Hệ điều hành:** Windows 10/11 (64-bit) hoặc Ubuntu 20.04/22.04 LTS.
- **Python:** Phiên bản 3.10 đến 3.14.
- **GPU (Khuyến nghị):** NVIDIA GeForce GTX 1650 / RTX 3050 trở lên (Hỗ trợ OpenGL 3.3+).

### Bước 1: Clone mã nguồn repository
```bash
git clone https://github.com/tranvanmanh9325/medical-science.git
cd medical-science
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements-train.txt
```

### Bước 3: Khởi chạy phòng thí nghiệm 3D
- **Cách 1 (Khuyến nghị trên Windows):** Nhấp đúp chuột vào file **`run.bat`** *(Tự động quét dọn tiến trình cũ, ngăn ngừa xung đột GPU và mở giao diện)*.
- **Cách 2:** Chạy trực tiếp qua terminal:
  ```powershell
  python main.py
  ```

### Bước 4: Chạy thử mẫu huấn luyện PPO cục bộ (Smoke Test)
```powershell
python test_mini_train_sample.py
```

---

## 📖 7. Tài Liệu Tham Khảo (References & Acknowledgements)

1. **Apptronik & Google DeepMind:** [MuJoCo Menagerie Apptronik Apollo Model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo).
2. **MuJoCo Physics Engine:** E. Todorov, T. Erez, and Y. Tassa, *"MuJoCo: A physics engine for model-based control,"* IROS 2012.
3. **MJX (MuJoCo XLA):** DeepMind Technologies, *"Hardware-Accelerated Physics Simulation in JAX."*
4. **PPO Algorithm:** J. Schulman et al., *"Proximal Policy Optimization Algorithms,"* arXiv:1707.06347, 2017.
5. **da Vinci Research Kit (dVRK):** Intuitive Surgical & Johns Hopkins University ERC CISST.

---

<div align="center">
  <b>Phát triển vì mục tiêu thúc đẩy Khoa học Robot Hình người & Cơ sinh học Y tế Việt Nam 🇻🇳</b>
</div>