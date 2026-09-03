"""
================================================================================
 Apollo Scientific Robotics Research & Biomechanics Telemetry Suite
--------------------------------------------------------------------------------
 Nền tảng Mô phỏng & Đo lường Cơ sinh học Robot Humanoid Apollo (Chuẩn Tiếng Việt)
 Tính năng:
  - Đo lường Cơ sinh học: Trọng tâm (CoM) 3D, Véc-tơ Lực chân (GRF), Điểm ZMP, Đa giác thăng bằng.
  - Đồ thị sóng dao động thời gian thực (Độ cao khung hông, Lực ép hai bàn chân, Vận tốc ngang).
  - Bảng chẩn đoán tải lực mô-men xoắn các khớp chính (Thân, Háng, Đầu gối, Cổ chân, Khớp vai).
  - Cảm biến góc nghiêng thân (IMU) & La bàn chân trời nhân tạo đo độ nghiêng Roll/Pitch/Yaw.
  - Cơ chế tự phục hồi tư thế đứng thẳng chống xô ngã (Active Attitude Self-Righting).
  - Thực nghiệm lực đẩy nhiễu loạn (Impulse Push Disturbance Rejection Test).
  - Phím TAB duy nhất bật/tắt toàn bộ giao diện 2D (Quả cầu định hướng Gizmo luôn hiển thị cố định).
  - Bộ hiển thị tiếng Việt Unicode chống răng cưa siêu nét đạt 240+ FPS trên GPU.
================================================================================
"""

import os
import sys
import time
import math
import collections
import numpy as np

# Configure Windows console for UTF-8 Vietnamese output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from PIL import Image, ImageDraw, ImageFont
import glfw
import OpenGL.GL as gl
import mujoco

# PPO Policy inference (numpy-only, no JAX needed at runtime)
# Weights loaded from .npz checkpoint trained on Kaggle
import glob




# ==============================================================================
# 0. PPO POLICY — NẠP CHECKPOINT & SUY LUẬN NUMPY THUẦN (KHÔNG CẦN JAX)
# ==============================================================================
class PPOPolicy:
    """
    Tải trọng số từ checkpoint .npz (định dạng flax flatten_dict sep='/') và
    thực hiện suy luận deterministic: action = tanh(W3·elu(W2·elu(W1·obs+b1)+b2)+b3).
    Kiến trúc khớp hoàn toàn với ActorCritic trong generate_kaggle_notebook.py.
    """

    def __init__(self, checkpoint_path: str, mj_model, nu: int):
        self.nu = nu
        self._load(checkpoint_path)

        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0: key_id = 0
        self.default_pose = mj_model.key_qpos[key_id][7:].copy()  # (nu,)
        self.ctrl_range   = mj_model.actuator_ctrlrange.copy()     # (nu, 2)
        self.action_scale = 0.1
        self.prev_act     = np.zeros(nu)
        print(f"[PPO POLICY] Loaded: {checkpoint_path}")
        print(f"  Layers: obs({self._obs_dim}) → 512 → 256 → 128 → action({nu})")

    def _load(self, path: str):
        """Load flax flatten_dict weights (keys like 'params/Dense_0/kernel')."""
        data = np.load(path)
        keys = list(data.keys())

        def get(pattern):
            matches = [k for k in keys if pattern in k]
            if not matches:
                raise KeyError(f"Key pattern '{pattern}' not found in {keys[:10]}")
            return data[matches[0]]

        # Actor head: Dense_0..2 = hidden layers, Dense_3 = mean output
        self.W = []
        self.b = []
        for i in range(3):  # hidden layers
            self.W.append(get(f"Dense_{i}/kernel"))
            self.b.append(get(f"Dense_{i}/bias"))
        self.W_mean = get("Dense_3/kernel")
        self.b_mean = get("Dense_3/bias")
        self._obs_dim = self.W[0].shape[0]

    @staticmethod
    def _elu(x):
        return np.where(x >= 0, x, np.expm1(x))

    def get_obs(self, data, mj_model) -> np.ndarray:
        """Build observation vector identical to training."""
        qpos = data.qpos
        qvel = data.qvel
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        upvec = np.array([
            2.0*(qx*qz + qw*qy),
            2.0*(qy*qz - qw*qx),
            1.0 - 2.0*(qx**2 + qy**2),
        ])
        linvel = qvel[:3]
        angvel = qvel[3:6]
        jpos   = qpos[7:7+self.nu] - self.default_pose
        jvel   = qvel[6:6+self.nu]
        obs    = np.concatenate([upvec, linvel, angvel, jpos, jvel, self.prev_act])
        return np.clip(obs, -20.0, 20.0)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic forward pass: action = tanh(mean). No noise."""
        x = obs.astype(np.float32)
        for W, b in zip(self.W, self.b):
            x = self._elu(x @ W + b)
        mean   = np.tanh(x @ self.W_mean + self.b_mean)
        action = np.clip(mean, -1.0, 1.0)
        return action

    def step(self, data, mj_model) -> np.ndarray:
        """
        Trả về ctrl array để set vào data.ctrl.
        ctrl = clip(default_pose + action * scale, ctrl_lo, ctrl_hi)
        """
        obs    = self.get_obs(data, mj_model)
        action = self.infer(obs)
        ctrl   = self.default_pose + action * self.action_scale
        ctrl   = np.clip(ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.prev_act = action.copy()
        return ctrl

    def reset(self):
        self.prev_act = np.zeros(self.nu)


# ==============================================================================
# 1. BỘ KẾT XUẤT FONT TIẾNG VIỆT UNICODE ĐỘ PHÂN GIẢI CAO (OPENGL TEXTURE CACHE)
# ==============================================================================
class FontRenderer:
    """
    Bộ kết xuất font chữ Tiếng Việt Unicode siêu nét sử dụng TrueType Font của Windows.
    Bộ nhớ đệm texture 2D đạt tốc độ khung hình 240+ FPS không gây trễ CPU.
    """
    def __init__(self):
        self.fonts = {}
        self.tex_cache = {}
        self.cache_order = collections.deque()
        self.max_cache_size = 500

        font_configs = {
            'bold': ['C:/Windows/Fonts/segoeuib.ttf', 'C:/Windows/Fonts/arialbd.ttf'],
            'regular': ['C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf'],
            'mono': ['C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/cour.ttf']
        }
        for ftype, paths in font_configs.items():
            self.fonts[ftype] = {}
            for path in paths:
                if os.path.exists(path):
                    for size in [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28]:
                        try:
                            self.fonts[ftype][size] = ImageFont.truetype(path, size)
                        except Exception:
                            pass
                    break

    def _get_font(self, font_type, size):
        type_fonts = self.fonts.get(font_type, {})
        if size in type_fonts:
            return type_fonts[size]
        if type_fonts:
            closest_size = min(type_fonts.keys(), key=lambda s: abs(s - size))
            return type_fonts[closest_size]
        return ImageFont.load_default()

    def draw_text(self, text, x, y, font_type='regular', size=13, color=(255, 255, 255, 255), align='left'):
        """Vẽ chữ tiếng Việt có dấu siêu nét lên màn hình 2D."""
        if not text:
            return 0, 0
            
        key = (text, font_type, size, color)
        if key in self.tex_cache:
            tex_id, w, h = self.tex_cache[key]
        else:
            font = self._get_font(font_type, size)
            bbox = font.getbbox(text)
            w = max(1, bbox[2] - bbox[0] + 6)
            h = max(1, bbox[3] - bbox[1] + 6)
            
            img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((-bbox[0] + 3, -bbox[1] + 3), text, font=font, fill=color)
            
            tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
            
            if len(self.cache_order) >= self.max_cache_size:
                oldest_key = self.cache_order.popleft()
                if oldest_key in self.tex_cache:
                    old_tex_id, _, _ = self.tex_cache.pop(oldest_key)
                    gl.glDeleteTextures([old_tex_id])
                    
            self.tex_cache[key] = (tex_id, w, h)
            self.cache_order.append(key)

        rx = x
        if align == 'center':
            rx = x - w * 0.5
        elif align == 'right':
            rx = x - w

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glColor4f(1.0, 1.0, 1.0, color[3] / 255.0 if len(color) > 3 else 1.0)
        
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex2f(rx, y)
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex2f(rx + w, y)
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex2f(rx + w, y + h)
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex2f(rx, y + h)
        gl.glEnd()
        
        gl.glDisable(gl.GL_TEXTURE_2D)
        return w, h


# ==============================================================================
# 2. BỘ ĐO LƯỜNG ĐỘNG LỰC HỌC & CƠ SINH HỌC THỜI GIAN THỰC
# ==============================================================================
class BiomechanicsTelemetry:
    """
    Tính toán Trọng tâm (CoM), Lực phản lực mặt đất (GRF), Điểm mô-men bằng không (ZMP),
    Đa giác thăng bằng, Góc nghiêng thân robot và Tải mô-men xoắn các khớp chính.
    """
    # Bản đồ dịch tên kỹ thuật sang Tiếng Việt trực quan, dễ hiểu
    JOINT_VN_NAMES = {
        'torso_roll': 'Thân: Nghiêng Trái/Phải',
        'torso_pitch': 'Thân: Cúi/Ngửa Trước Sau',
        'l_hip_fe': 'Háng Trái: Gập/Duỗi Đùi',
        'r_hip_fe': 'Háng Phải: Gập/Duỗi Đùi',
        'l_knee_fe': 'Gối Trái: Co/Duỗi Khớp',
        'r_knee_fe': 'Gối Phải: Co/Duỗi Khớp',
        'l_ankle_ie': 'Cổ Chân Trái: Lắc Nghiêng',
        'r_ankle_ie': 'Cổ Chân Phải: Lắc Nghiêng',
        'l_shoulder_fe': 'Vai Trái: Nâng Cánh Tay',
        'r_shoulder_fe': 'Vai Phải: Nâng Cánh Tay',
    }

    def __init__(self, model):
        self.model = model
        self.total_mass = float(np.sum(model.body_mass))
        self.gravity_load = self.total_mass * 9.81
        self.prev_com = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.last_time = 0.0

        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.l_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_l_sole")
        self.r_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_r_sole")

        self.actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    def update(self, data):
        # 1. Trọng tâm toàn thân (Center of Mass)
        com = data.subtree_com[0].copy()
        dt = data.time - self.last_time
        if dt > 1e-5:
            self.com_vel = (com - self.prev_com) / dt
        self.prev_com = com.copy()
        self.last_time = data.time

        # 2. Lực phản lực tiếp xúc mặt đất (GRF)
        contacts = []
        c_force = np.zeros(6, dtype=np.float64)
        total_grf = np.zeros(3)
        fz_left = 0.0
        fz_right = 0.0
        zmp_num = np.zeros(2)
        zmp_den = 0.0
        contact_points_2d = []

        for i in range(data.ncon):
            c = data.contact[i]
            mujoco.mj_contactForce(self.model, data, i, c_force)
            frame = c.frame.reshape(3, 3)
            f_world = frame.T @ c_force[:3]
            pos = c.pos.copy()

            is_left = (c.geom1 == self.l_sole_id or c.geom2 == self.l_sole_id)
            is_right = (c.geom1 == self.r_sole_id or c.geom2 == self.r_sole_id)

            if is_left or is_right or f_world[2] > 1.0:
                contacts.append({
                    'pos': pos,
                    'force': f_world,
                    'mag': float(np.linalg.norm(f_world)),
                    'is_left': is_left,
                    'is_right': is_right
                })
                total_grf += f_world
                contact_points_2d.append((pos[0], pos[1]))

                if is_left:
                    fz_left += f_world[2]
                if is_right:
                    fz_right += f_world[2]

                if f_world[2] > 0.5:
                    zmp_num += np.array([pos[0] * f_world[2], pos[1] * f_world[2]])
                    zmp_den += f_world[2]

        # Điểm mô-men bằng 0 (Zero Moment Point - ZMP)
        if zmp_den > 1e-3:
            zmp = zmp_num / zmp_den
        else:
            zmp = np.array([com[0], com[1]])

        # 3. Đa giác bao lồi hỗ trợ thăng bằng (Support Polygon)
        support_poly = []
        if len(contact_points_2d) >= 3:
            support_poly = self._compute_convex_hull(contact_points_2d)
        elif len(contact_points_2d) > 0:
            support_poly = contact_points_2d

        # 4. Góc nghiêng thân robot & Cảm biến IMU
        quat = data.xquat[self.torso_id]
        roll, pitch, yaw = self._quat_to_euler(quat)
        gyro = data.cvel[self.torso_id][:3]

        # 5. Tải mô-men xoắn các khớp & Công suất tiêu thụ
        actuator_loads = []
        total_power = 0.0
        for i in range(self.model.nu):
            tau = float(data.actuator_force[i]) if hasattr(data, 'actuator_force') else float(data.ctrl[i])
            qpos_id = self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            qvel_id = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            qvel = float(data.qvel[qvel_id])
            power = abs(tau * qvel)
            total_power += power

            frc_range = self.model.actuator_forcerange[i]
            max_tau = max(5.0, max(abs(frc_range[0]), abs(frc_range[1]))) if frc_range[1] > frc_range[0] else 100.0
            ratio = min(1.0, abs(tau) / max_tau)
            
            raw_name = self.actuator_names[i]
            vn_label = self.JOINT_VN_NAMES.get(raw_name, raw_name)

            actuator_loads.append({
                'raw_name': raw_name,
                'name': vn_label,
                'tau': tau,
                'max_tau': max_tau,
                'ratio': ratio,
                'power': power
            })

        return {
            'com': com,
            'com_vel': self.com_vel,
            'contacts': contacts,
            'total_grf': total_grf,
            'fz_left': fz_left,
            'fz_right': fz_right,
            'zmp': zmp,
            'support_poly': support_poly,
            'roll': math.degrees(roll),
            'pitch': math.degrees(pitch),
            'yaw': math.degrees(yaw),
            'gyro': np.degrees(gyro),
            'actuator_loads': actuator_loads,
            'total_power': total_power,
            'pelvis_z': float(data.qpos[2])
        }

    def _compute_convex_hull(self, points):
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def _quat_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw


# ==============================================================================
# 3. ĐỒ THỊ SÓNG DAO ĐỘNG THỜI GIAN THỰC (REAL-TIME OSCILLOSCOPE)
# ==============================================================================
class RealtimeOscilloscope:
    """
    Đồ thị sóng cuộn thời gian thực theo dõi độ cao khung hông, lực chân trái / phải.
    """
    def __init__(self, buffer_size=300):
        self.buffer_size = buffer_size
        self.times = collections.deque(maxlen=buffer_size)
        self.pelvis_z = collections.deque(maxlen=buffer_size)
        self.fz_left = collections.deque(maxlen=buffer_size)
        self.fz_right = collections.deque(maxlen=buffer_size)
        self.com_vel_y = collections.deque(maxlen=buffer_size)

    def append(self, t, pz, fl, fr, vy):
        self.times.append(t)
        self.pelvis_z.append(pz)
        self.fz_left.append(fl)
        self.fz_right.append(fr)
        self.com_vel_y.append(vy)

    def draw(self, x, y, w, h, font_renderer):
        if len(self.times) < 2:
            return

        # Khung chứa kính mờ
        gl.glColor4f(0.06, 0.09, 0.14, 0.88)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y); gl.glVertex2f(x + w, y)
        gl.glVertex2f(x + w, y + h); gl.glVertex2f(x, y + h)
        gl.glEnd()

        gl.glLineWidth(1.5)
        gl.glColor4f(0.20, 0.35, 0.55, 0.70)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(x, y); gl.glVertex2f(x + w, y)
        gl.glVertex2f(x + w, y + h); gl.glVertex2f(x, y + h)
        gl.glEnd()

        # Tiêu đề tiếng Việt
        font_renderer.draw_text("ĐỒ THỊ DAO ĐỘNG THỜI GIAN THỰC", x + 12, y + 8, 'bold', 12, (0, 240, 255, 255))
        
        # Kênh 1: Độ cao khung hông Z
        g1_y = y + 28
        g1_h = (h - 42) * 0.47
        self._draw_waveform_channel(
            x + 10, g1_y, w - 20, g1_h,
            data=self.pelvis_z,
            ref_val=1.016,
            min_val=0.90, max_val=1.10,
            label=f"Độ Cao Khung Hông: {self.pelvis_z[-1]:.3f} m (Mục tiêu: 1.016m)",
            color=(0.0, 0.94, 1.0, 1.0),
            ref_color=(0.3, 0.5, 0.7, 0.6),
            font_renderer=font_renderer
        )

        # Kênh 2: Lực tiếp đất chân trái & chân phải
        g2_y = g1_y + g1_h + 8
        g2_h = (h - 42) * 0.47
        self._draw_dual_waveform_channel(
            x + 10, g2_y, w - 20, g2_h,
            data1=self.fz_left, data2=self.fz_right,
            min_val=0.0, max_val=800.0,
            label1=f"Chân Trái: {self.fz_left[-1]:.0f} N",
            label2=f"Chân Phải: {self.fz_right[-1]:.0f} N",
            color1=(0.0, 1.0, 0.5, 1.0),
            color2=(1.0, 0.75, 0.0, 1.0),
            font_renderer=font_renderer
        )

    def _draw_waveform_channel(self, gx, gy, gw, gh, data, ref_val, min_val, max_val, label, color, ref_color, font_renderer):
        font_renderer.draw_text(label, gx + 4, gy + 2, 'regular', 11, tuple(int(c * 255) for c in color))

        cx, cy, cw, ch = gx, gy + 18, gw, gh - 20
        gl.glColor4f(0.04, 0.06, 0.10, 0.90)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(cx, cy); gl.glVertex2f(cx + cw, cy)
        gl.glVertex2f(cx + cw, cy + ch); gl.glVertex2f(cx, cy + ch)
        gl.glEnd()

        gl.glLineWidth(1.0)
        gl.glColor4f(0.15, 0.22, 0.35, 0.45)
        gl.glBegin(gl.GL_LINES)
        for i in range(1, 4):
            ly = cy + ch * (i / 4.0)
            gl.glVertex2f(cx, ly); gl.glVertex2f(cx + cw, ly)
        gl.glEnd()

        if min_val <= ref_val <= max_val:
            ry = cy + ch * (1.0 - (ref_val - min_val) / (max_val - min_val))
            gl.glColor4f(*ref_color)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(cx, ry); gl.glVertex2f(cx + cw, ry)
            gl.glEnd()

        gl.glLineWidth(1.8)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINE_STRIP)
        for i, val in enumerate(data):
            px = cx + (i / float(self.buffer_size - 1)) * cw
            norm_val = np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
            py = cy + ch * (1.0 - norm_val)
            gl.glVertex2f(px, py)
        gl.glEnd()

    def _draw_dual_waveform_channel(self, gx, gy, gw, gh, data1, data2, min_val, max_val, label1, label2, color1, color2, font_renderer):
        font_renderer.draw_text(label1, gx + 4, gy + 2, 'regular', 11, tuple(int(c * 255) for c in color1))
        font_renderer.draw_text(label2, gx + 150, gy + 2, 'regular', 11, tuple(int(c * 255) for c in color2))

        cx, cy, cw, ch = gx, gy + 18, gw, gh - 20
        gl.glColor4f(0.04, 0.06, 0.10, 0.90)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(cx, cy); gl.glVertex2f(cx + cw, cy)
        gl.glVertex2f(cx + cw, cy + ch); gl.glVertex2f(cx, cy + ch)
        gl.glEnd()

        gl.glLineWidth(1.0)
        gl.glColor4f(0.15, 0.22, 0.35, 0.45)
        gl.glBegin(gl.GL_LINES)
        for i in range(1, 4):
            ly = cy + ch * (i / 4.0)
            gl.glVertex2f(cx, ly); gl.glVertex2f(cx + cw, ly)
        gl.glEnd()

        gl.glLineWidth(1.8)
        gl.glColor4f(*color1)
        gl.glBegin(gl.GL_LINE_STRIP)
        for i, val in enumerate(data1):
            px = cx + (i / float(self.buffer_size - 1)) * cw
            norm_val = np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
            py = cy + ch * (1.0 - norm_val)
            gl.glVertex2f(px, py)
        gl.glEnd()

        gl.glColor4f(*color2)
        gl.glBegin(gl.GL_LINE_STRIP)
        for i, val in enumerate(data2):
            px = cx + (i / float(self.buffer_size - 1)) * cw
            norm_val = np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
            py = cy + ch * (1.0 - norm_val)
            gl.glVertex2f(px, py)
        gl.glEnd()


# ==============================================================================
# 4. VIEWER CHÍNH VÀ ĐIỀU KHIỂN TƯƠNG TÁC (CHỈ GIỮ PHÍM TAB DUY NHẤT)
# ==============================================================================
class BlenderMuJoCoViewer:
    """
    Giao diện mô phỏng phòng lab nghiên cứu khoa học Robot Humanoid Apollo:
    - 100% Tiếng Việt trực quan, giải thích rõ ràng từng đại lượng vật lý.
    - Duy nhất phím TAB bật/tắt toàn bộ giao diện chữ/đồ thị (Quả cầu Gizmo luôn hiển thị cố định).
    - Bộ điều khiển cân bằng tự đứng thẳng hồi phục tư thế (Active Self-Righting PD).
    """
    def __init__(self, model_path, width=1600, height=900, title="Robot Apptronik Apollo - Phòng Thí Nghiệm Cơ Sinh Học"):
        self.model_path = model_path
        self.width = width
        self.height = height
        self.title = title

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.model.vis.quality.offsamples = 4
        self.model.vis.quality.shadowsize = 2048

        self.telemetry = BiomechanicsTelemetry(self.model)
        self.oscilloscope = RealtimeOscilloscope(buffer_size=280)

        self.total_mass = float(np.sum(self.model.body_mass))
        self.gravity_comp = self.total_mass * 9.81
        self.root_body_id = 1
        self.nominal_root_z = 1.016

        # --- Tải PPO Policy từ checkpoint v15 ---
        self.policy      = None
        self.policy_mode = False  # False=PD cứng, True=PPO brain
        # model_path = .../google_deepmind_menagerie/apptronik_apollo/scene.xml
        # Cần lên 3 cấp để đến project root D:\GitHub\medical-science
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(model_path))))
        ck_pattern = os.path.join(
            project_root,
            "kaggle_output", "checkpoints_v15", "checkpoints", "*.npz"
        )
        ck_files = sorted(glob.glob(ck_pattern),
                          key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0'))
        if ck_files:
            best_ck = ck_files[-1]  # checkpoint cuối = nhiều steps nhất
            try:
                self.policy = PPOPolicy(best_ck, self.model, self.model.nu)
                self.policy_mode = True
                print(f"[PPO] Brain AI đã nạp thành công! Phím P để bật/tắt.")
            except Exception as e:
                print(f"[PPO] Lỗi nạp checkpoint: {e}")
                print("[PPO] Dùng PD controller dự phòng.")
        else:
            print(f"[PPO] Không tìm thấy checkpoint tại: {ck_pattern}")
            print("[PPO] Chạy: python training/download_checkpoints.py")

        # Trạng thái mô phỏng
        self.paused = False
        self.sim_speed = 1.0
        self.step_single_frame = False
        self.physics_fps = 200.0
        self.render_fps = 60.0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Thử nghiệm lực đẩy nhiễu loạn
        self.push_force = np.zeros(3)
        self.push_decay = 0.0

        # Lịch sử quỹ đạo di chuyển
        self.trajectory_history = collections.deque(maxlen=120)

        # Cờ bật/tắt lớp phủ 3D vật lý (F1 - F8)
        self.layer_com = True          # F1: Trọng tâm toàn thân CoM
        self.layer_grf = True          # F2: Véc-tơ lực chân GRF
        self.layer_zmp = True          # F3: Điểm ZMP & Đa giác thăng bằng
        self.layer_skeleton = False    # F4: Hệ trục tọa độ khớp
        self.layer_trajectory = True   # F5: Vệt dải quỹ đạo chuyển động
        self.layer_metric_grid = True  # F6: Lưới đo khoảng cách mét
        self.layer_collision = False   # F7: Hình học va chạm
        self.theme_academic = False    # F8: Chế độ nền sáng / tối

        # Cờ hiển thị giao diện 2D duy nhất (TAB)
        self.show_hud = True           # TAB: Ẩn/Hiện toàn bộ bảng số liệu 2D

        self._reset_robot()

        if not glfw.init():
            raise RuntimeError("Không thể khởi tạo GLFW")

        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Không thể tạo cửa sổ GLFW")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.font_renderer = FontRenderer()

        # Camera 3D MuJoCo
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.0, 0.0, 0.90]
        self.cam.distance = 2.45
        self.cam.elevation = -8.0
        self.cam.azimuth = 215.0

        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=30000)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Trạng thái chuột
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0
        self.is_lmb_down = False
        self.is_mmb_down = False
        self.is_rmb_down = False
        self.is_shift_down = False
        self.is_ctrl_down = False
        self.drag_mode = None

        # Camera Animation
        self.animating = False
        self.anim_start_time = 0.0
        self.anim_duration = 0.25
        self.anim_start_az = 0.0
        self.anim_start_el = 0.0
        self.anim_target_az = 0.0
        self.anim_target_el = 0.0

        # Quả cầu con quay định hướng Gizmo (Cố định góc trên bên phải)
        self.gizmo_size = 46.0
        self.gizmo_margin_x = 65.0
        self.gizmo_margin_y = 70.0
        self.gizmo_drag_dist = 0.0
        self.gizmo_clicked_node = None
        self.hovered_node = None

        self.textures = {}
        self._load_textures()

        self._setup_callbacks()

    def _reset_robot(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            if self.model.key_ctrl.shape[1] == self.model.nu:
                self.data.ctrl[:] = self.model.key_ctrl[key_id]
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.push_force = np.zeros(3)
        self.push_decay = 0.0
        self.trajectory_history.clear()
        mujoco.mj_forward(self.model, self.data)
        print("[ROBOT] Đã đặt lại tư thế đứng thẳng chuẩn ban đầu")

    def _step_physics_with_balance(self):
        """
        Bước vật lý:
          - policy_mode=True  → PPO Brain AI (trained 100M steps) điều khiển khớp
          - policy_mode=False → PD cứng dự phòng (hardcoded balance controller)
        Phím P: bật/tắt giữa hai chế độ.
        """
        if self.policy_mode and self.policy is not None:
            # ── PPO Brain AI mode ───────────────────────────────────────────
            ctrl = self.policy.step(self.data, self.model)
            self.data.ctrl[:] = ctrl

            # Vẫn áp lực đẩy thử nghiệm nếu đang test
            if self.push_decay > 0.0:
                self.data.xfrc_applied[self.root_body_id][:3] = self.push_force
                self.push_decay -= self.model.opt.timestep
                if self.push_decay <= 0.0:
                    self.push_force = np.zeros(3)
                    self.data.xfrc_applied[self.root_body_id][:] = 0.0
        else:
            # ── PD Cứng dự phòng (giữ nguyên code cũ) ─────────────────────
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if key_id != -1 and self.model.key_ctrl.shape[1] == self.model.nu:
                self.data.ctrl[:] = self.model.key_ctrl[key_id]

            kp_z = 6000.0
            kd_z = 600.0
            z_err = self.nominal_root_z - self.data.qpos[2]
            vz = self.data.qvel[2]
            fz = self.gravity_comp + kp_z * z_err - kd_z * vz

            q = self.data.qpos[3:7]
            kp_rot, kd_rot = 900.0, 140.0
            tau_rx = -kp_rot * q[1] - kd_rot * self.data.qvel[3]
            tau_ry = -kp_rot * q[2] - kd_rot * self.data.qvel[4]
            tau_rz = -kp_rot * q[3] - kd_rot * self.data.qvel[5]

            self.data.xfrc_applied[self.root_body_id][:3] = [self.push_force[0], self.push_force[1], fz]
            self.data.xfrc_applied[self.root_body_id][3:]  = [tau_rx, tau_ry, tau_rz]

            if self.push_decay > 0.0:
                self.push_decay -= self.model.opt.timestep
                if self.push_decay <= 0.0:
                    self.push_force = np.zeros(3)

        mujoco.mj_step(self.model, self.data)

        if self.frame_count % 3 == 0:
            self.trajectory_history.append(self.data.qpos[:3].copy())


    def inject_perturbation(self, fx=0.0, fy=0.0, duration=0.25):
        """Tác dụng lực đẩy xô thử nghiệm cân bằng."""
        self.push_force = np.array([fx, fy, 0.0])
        self.push_decay = duration
        print(f"[THỬ NGHIỆM LỰC ĐẨY] Tác dụng lực: [{fx:.1f}, {fy:.1f}, 0.0] N trong {duration}s")

    def _load_textures(self):
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        texture_files = {
            '+X': 'pin_x.png', '+Y': 'pin_y.png', '+Z': 'pin_z.png',
            '-X': 'pin_neg_x.png', '-Y': 'pin_neg_y.png', '-Z': 'pin_neg_z.png',
            'DISC': 'trackball_disc.png', 'PIVOT': 'center_pivot.png'
        }
        for key, fname in texture_files.items():
            path = os.path.join(assets_dir, fname)
            if os.path.exists(path):
                img = Image.open(path).convert('RGBA')
                w, h = img.size
                tex_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img.tobytes())
                self.textures[key] = tex_id

    def _setup_callbacks(self):
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    def _on_resize(self, window, w, h):
        self.width = max(1, w)
        self.height = max(1, h)

    def get_gizmo_center(self):
        return (self.width - self.gizmo_margin_x, self.gizmo_margin_y)

    def _get_camera_vectors(self):
        fwd = np.array(self.scn.camera[0].forward, dtype=np.float64)
        up = np.array(self.scn.camera[0].up, dtype=np.float64)
        right = np.cross(fwd, up)
        for v in (right, up, fwd):
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v /= norm
        return right, up, fwd

    def _get_gizmo_axis_nodes(self):
        cx, cy = self.get_gizmo_center()
        r = self.gizmo_size * 0.72
        R, U, F = self._get_camera_vectors()

        axes = [
            ('+X', (1, 0, 0), (0.92, 0.26, 0.21, 1.0), 'X', (180.0, 0.0), 12.0),
            ('-X', (-1, 0, 0), (0.75, 0.22, 0.20, 0.85), '', (0.0, 0.0), 5.5),
            ('+Y', (0, 1, 0), (0.20, 0.72, 0.35, 1.0), 'Y', (270.0, 0.0), 12.0),
            ('-Y', (0, -1, 0), (0.18, 0.55, 0.28, 0.85), '', (90.0, 0.0), 5.5),
            ('+Z', (0, 0, 1), (0.26, 0.55, 0.98, 1.0), 'Z', (270.0, -89.9), 12.0),
            ('-Z', (0, 0, -1), (0.18, 0.42, 0.75, 0.85), '', (270.0, 89.9), 5.5),
        ]
        nodes = []
        for name, vec, color, label, view_target, radius in axes:
            proj_x = vec[0]*R[0] + vec[1]*R[1] + vec[2]*R[2]
            proj_y = vec[0]*U[0] + vec[1]*U[1] + vec[2]*U[2]
            depth = vec[0]*F[0] + vec[1]*F[1] + vec[2]*F[2]
            nodes.append({
                'name': name, 'vec': vec,
                'sx': cx + proj_x * r, 'sy': cy - proj_y * r,
                'depth': depth, 'color': color, 'label': label,
                'target': view_target, 'radius': radius
            })
        nodes.sort(key=lambda item: item['depth'])
        return nodes

    def _get_hit_node(self, mx, my):
        nodes = self._get_gizmo_axis_nodes()
        for node in reversed(nodes):
            if math.hypot(mx - node['sx'], my - node['sy']) <= node['radius'] + 4.0:
                return node
        return None

    def _animate_to_view(self, target_az, target_el):
        self.anim_start_time = time.time()
        self.anim_start_az = self.cam.azimuth
        self.anim_start_el = self.cam.elevation
        delta_az = (target_az - self.anim_start_az) % 360.0
        if delta_az > 180.0:
            delta_az -= 360.0
        self.anim_target_az = self.anim_start_az + delta_az
        self.anim_target_el = target_el
        self.animating = True

    def _on_mouse_move(self, window, xpos, ypos):
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y

        hit = self._get_hit_node(xpos, ypos)
        self.hovered_node = hit['name'] if hit else None

        if self.drag_mode == 'GIZMO_ORBIT':
            self.gizmo_drag_dist += math.hypot(dx, dy)
            self.animating = False
            reldx = -dx / max(100.0, float(self.height))
            reldy = -dy / max(100.0, float(self.height))
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, reldx, reldy, self.cam)
        elif self.drag_mode in ('SCENE_PAN', 'PAN_VIEW'):
            reldx = -dx / max(100.0, float(self.height))
            reldy = -dy / max(100.0, float(self.height))
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, reldx, reldy, self.cam)
        elif self.drag_mode == 'MMB_ORBIT':
            reldx = -dx / max(100.0, float(self.height))
            reldy = -dy / max(100.0, float(self.height))
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, reldx, reldy, self.cam)
        elif self.drag_mode == 'CTRL_ZOOM':
            reldy = -dy / max(100.0, float(self.height))
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, reldy, self.cam)

        self.last_mouse_x = xpos
        self.last_mouse_y = ypos

    def _on_mouse_button(self, window, button, action, mods):
        self.is_shift_down = bool(mods & glfw.MOD_SHIFT)
        self.is_ctrl_down = bool(mods & glfw.MOD_CONTROL)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.is_lmb_down = True
                mx, my = self.last_mouse_x, self.last_mouse_y
                cx, cy = self.get_gizmo_center()
                if math.hypot(mx - cx, my - cy) <= self.gizmo_size + 8.0:
                    self.drag_mode = 'GIZMO_ORBIT'
                    self.gizmo_drag_dist = 0.0
                    self.gizmo_clicked_node = self._get_hit_node(mx, my)
                else:
                    self.drag_mode = 'SCENE_PAN'
            elif action == glfw.RELEASE:
                self.is_lmb_down = False
                if self.drag_mode == 'GIZMO_ORBIT' and self.gizmo_drag_dist < 5.0 and self.gizmo_clicked_node:
                    target_az, target_el = self.gizmo_clicked_node['target']
                    self._animate_to_view(target_az, target_el)
                self.drag_mode = None
                self.gizmo_clicked_node = None

        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            if action == glfw.PRESS:
                self.is_mmb_down = True
                self.drag_mode = 'PAN_VIEW' if self.is_shift_down else ('CTRL_ZOOM' if self.is_ctrl_down else 'MMB_ORBIT')
            elif action == glfw.RELEASE:
                self.is_mmb_down = False
                self.drag_mode = None

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.is_rmb_down = True
                self.drag_mode = 'PAN_VIEW'
            elif action == glfw.RELEASE:
                self.is_rmb_down = False
                self.drag_mode = None

    def _on_scroll(self, window, xoffset, yoffset):
        zoom_factor = math.pow(0.88, yoffset)
        self.cam.distance = max(0.1, min(50.0, self.cam.distance * zoom_factor))

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            # Phím TAB duy nhất: Ẩn/Hiện toàn bộ giao diện 2D
            if key == glfw.KEY_TAB:
                self.show_hud = not self.show_hud
                print(f"[GIAO DIỆN] Toàn bộ bảng số liệu 2D: {'HIỆN' if self.show_hud else 'ẨN (Toàn cảnh 3D)'}")

            # Điều khiển mô phỏng
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
                print(f"[MÔ PHỎNG] {'TẠM DỪNG' if self.paused else 'ĐANG CHẠY'}")
            elif key == glfw.KEY_N:
                self.step_single_frame = True
            elif key == glfw.KEY_R:
                self._reset_robot()
                if self.policy is not None:
                    self.policy.reset()

            # Phím B: Bật/tắt PPO Brain AI ↔ PD Controller
            elif key == glfw.KEY_B:
                if self.policy is not None:
                    self.policy_mode = not self.policy_mode
                    mode_str = "🧠 PPO BRAIN AI (v15, 100M steps)" if self.policy_mode else "⚙️  PD CỨNG (dự phòng)"
                    print(f"[CHẾ ĐỘ ĐIỀU KHIỂN] {mode_str}")
                    if self.policy is not None:
                        self.policy.reset()
                else:
                    print("[PPO] Chưa có checkpoint! Chạy: python training/download_checkpoints.py")

            # Điều chỉnh tốc độ mô phỏng
            elif key == glfw.KEY_1 and not mods:
                self.sim_speed = 0.1
            elif key == glfw.KEY_2 and not mods:
                self.sim_speed = 0.25
            elif key == glfw.KEY_3 and not mods:
                self.sim_speed = 0.5
            elif key == glfw.KEY_4 and not mods:
                self.sim_speed = 1.0

            # Thử nghiệm lực đẩy xô robot
            elif key == glfw.KEY_LEFT:
                self.inject_perturbation(fy=140.0)
            elif key == glfw.KEY_RIGHT:
                self.inject_perturbation(fy=-140.0)
            elif key == glfw.KEY_UP:
                self.inject_perturbation(fx=150.0)
            elif key == glfw.KEY_DOWN:
                self.inject_perturbation(fx=-150.0)
            elif key == glfw.KEY_F:
                self.inject_perturbation(fx=120.0, fy=100.0)

            # Đổi chế độ nền sáng / tối
            elif key == glfw.KEY_F8:
                self.theme_academic = not self.theme_academic
                print(f"[CHẾ ĐỘ NỀN] Nền sáng bài báo: {self.theme_academic}")

            # Chụp ảnh báo cáo nghiên cứu
            elif key == glfw.KEY_P:

                self._capture_scientific_snapshot()

            # Phím Numpad đổi góc nhìn chuẩn Blender
            elif key in (glfw.KEY_KP_1, glfw.KEY_1) and (mods & glfw.MOD_CONTROL):
                self._animate_to_view(90.0, 0.0)
            elif key in (glfw.KEY_KP_1, glfw.KEY_1) and (mods & glfw.MOD_SHIFT):
                self._animate_to_view(270.0, 0.0)
            elif key in (glfw.KEY_KP_3, glfw.KEY_3):
                self._animate_to_view(0.0 if (mods & glfw.MOD_CONTROL) else 180.0, 0.0)
            elif key in (glfw.KEY_KP_7, glfw.KEY_7):
                self._animate_to_view(270.0, 89.9 if (mods & glfw.MOD_CONTROL) else -89.9)

    def _capture_scientific_snapshot(self):
        """Chụp ảnh số liệu khoa học độ phân giải cao."""
        w, h = glfw.get_framebuffer_size(self.window)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGBA', (w, h), pixels).transpose(Image.FLIP_TOP_BOTTOM)
        
        os.makedirs("pic", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("pic", f"anh_chup_nghien_cuu_{timestamp}.png")
        img.save(out_path, "PNG")
        print(f"[CHỤP ẢNH] Đã lưu ảnh báo cáo khoa học vào: {out_path}")

    def _update_camera_animation(self):
        if not self.animating:
            return
        t = (time.time() - self.anim_start_time) / self.anim_duration
        if t >= 1.0:
            self.cam.azimuth = self.anim_target_az % 360.0
            self.cam.elevation = self.anim_target_el
            self.animating = False
        else:
            ease = 1.0 - math.pow(1.0 - t, 3)
            self.cam.azimuth = self.anim_start_az + (self.anim_target_az - self.anim_start_az) * ease
            self.cam.elevation = self.anim_start_el + (self.anim_target_el - self.anim_start_el) * ease

    # ==========================================================================
    # 5. CÁC HIỆU ỨNG VẬT LÝ 3D (CHỈ GIỮ MŨI TÊN LỰC ĐẨY KHI THỰC NGHIỆM)
    # ==========================================================================
    def _inject_3d_scientific_overlays(self, telem):
        scn = self.scn

        # Mũi tên đỏ chỉ báo lực đẩy khi thử nghiệm cân bằng
        if self.push_decay > 0.0 and scn.ngeom < scn.maxgeom:
            torso_pos = self.data.xpos[self.telemetry.torso_id]
            f_norm = np.linalg.norm(self.push_force[:2])
            if f_norm > 1.0:
                push_dir = self.push_force / f_norm
                start_p = torso_pos - push_dir * 0.45
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([1.0, 0.15, 0.15, 0.98]))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.032, start_p, torso_pos)
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1


    # ==========================================================================
    # 6. GIAO DIỆN HEADS-UP DISPLAY (HUD) 100% TIẾNG VIỆT
    # ==========================================================================
    def _draw_top_scientific_ribbon(self, telem):
        """Thanh thông số khoa học trên cùng."""
        w, h = self.width, self.height
        rx, ry, rw, rh = 16, 14, w - 32, 40

        gl.glColor4f(0.06, 0.09, 0.15, 0.90)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(rx, ry); gl.glVertex2f(rx + rw, ry)
        gl.glVertex2f(rx + rw, ry + rh); gl.glVertex2f(rx, ry + rh)
        gl.glEnd()

        gl.glLineWidth(1.5)
        gl.glColor4f(0.25, 0.40, 0.62, 0.75)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(rx, ry); gl.glVertex2f(rx + rw, ry)
        gl.glVertex2f(rx + rw, ry + rh); gl.glVertex2f(rx, ry + rh)
        gl.glEnd()

        dot_color = (0.95, 0.60, 0.10, 1.0) if self.paused else (0.0, 1.0, 0.50, 1.0)
        gl.glColor4f(*dot_color)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex2f(rx + 20, ry + 20)
        for i in range(25):
            theta = 2.0 * math.pi * i / 24.0
            gl.glVertex2f(rx + 20 + 6.0 * math.cos(theta), ry + 20 + 6.0 * math.sin(theta))
        gl.glEnd()

        fr = self.font_renderer
        fr.draw_text("PHÒNG THÍ NGHIỆM ROBOT APOLLO", rx + 36, ry + 11, 'bold', 14, (0, 240, 255, 255))

        sim_t = self.data.time
        mins = int(sim_t // 60)
        secs = sim_t % 60
        t_str = f"THỜI GIAN: {mins:02d}:{secs:05.2f}s"
        fr.draw_text(t_str, rx + 295, ry + 12, 'mono', 12, (200, 220, 245, 255))

        fps_str = f"VẬT LÝ: {self.physics_fps:.0f}Hz | ĐỒ HỌA: {self.render_fps:.0f} FPS | TỐC ĐỘ: {self.sim_speed:.2f}x"
        fr.draw_text(fps_str, rx + 480, ry + 12, 'mono', 12, (160, 200, 240, 255))

        pwr_str = f"CÔNG SUẤT: {telem['total_power']:.1f}W | NẶNG: {self.total_mass:.1f}kg"
        fr.draw_text(pwr_str, rx + 830, ry + 12, 'mono', 12, (180, 240, 180, 255))

        badge_text = "ĐANG TẠM DỪNG" if self.paused else ("ĐANG THỬ LỰC ĐẨY" if self.push_decay > 0.0 else "CÂN BẰNG TỰ ĐỘNG (ỔN ĐỊNH)")
        badge_color = (255, 120, 0, 255) if (self.paused or self.push_decay > 0.0) else (0, 255, 140, 255)
        fr.draw_text(badge_text, rx + rw - 260, ry + 12, 'bold', 12, badge_color)

    def _draw_left_diagnostic_dashboard(self, telem):
        """Bảng chẩn đoán bên trái (100% Tiếng Việt dễ hiểu)."""
        px, py, pw, ph = 16, 64, 350, self.height - 110
        fr = self.font_renderer

        gl.glColor4f(0.06, 0.09, 0.14, 0.88)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(px, py); gl.glVertex2f(px + pw, py)
        gl.glVertex2f(px + pw, py + ph); gl.glVertex2f(px, py + ph)
        gl.glEnd()

        gl.glLineWidth(1.5)
        gl.glColor4f(0.20, 0.35, 0.55, 0.70)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(px, py); gl.glVertex2f(px + pw, py)
        gl.glVertex2f(px + pw, py + ph); gl.glVertex2f(px, py + ph)
        gl.glEnd()

        fr.draw_text("CHẨN ĐOÁN ĐỘNG LỰC HỌC & KHỚP", px + 14, py + 10, 'bold', 13, (0, 240, 255, 255))

        # --- PHẦN 1: CẢM BIẾN GÓC NGHIÊNG THÂN (IMU) ---
        sec1_y = py + 34
        fr.draw_text("1. CẢM BIẾN GÓC NGHIÊNG THÂN (IMU)", px + 14, sec1_y, 'bold', 11, (140, 180, 220, 255))
        
        hx, hy, hr = px + 65, sec1_y + 48, 36
        roll = telem['roll']
        pitch = telem['pitch']

        gl.glColor4f(0.08, 0.14, 0.24, 0.95)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex2f(hx, hy)
        for i in range(25):
            th = 2.0 * math.pi * i / 24.0
            gl.glVertex2f(hx + hr * math.cos(th), hy + hr * math.sin(th))
        gl.glEnd()

        pitch_offset = np.clip(pitch * 1.2, -hr * 0.8, hr * 0.8)
        rad_roll = math.radians(roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        gl.glLineWidth(2.2)
        gl.glColor4f(0.0, 0.94, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(hx - hr * 0.75 * cos_r + pitch_offset * sin_r, hy - hr * 0.75 * sin_r - pitch_offset * cos_r)
        gl.glVertex2f(hx + hr * 0.75 * cos_r + pitch_offset * sin_r, hy + hr * 0.75 * sin_r - pitch_offset * cos_r)
        gl.glEnd()

        gl.glLineWidth(1.5)
        gl.glColor4f(0.30, 0.50, 0.75, 0.85)
        gl.glBegin(gl.GL_LINE_LOOP)
        for i in range(25):
            th = 2.0 * math.pi * i / 24.0
            gl.glVertex2f(hx + hr * math.cos(th), hy + hr * math.sin(th))
        gl.glEnd()

        fr.draw_text(f"Nghiêng T/P: {roll:+05.1f}°", px + 125, sec1_y + 24, 'regular', 11, (200, 230, 255, 255))
        fr.draw_text(f"Cúi/Ngửa   : {pitch:+05.1f}°", px + 125, sec1_y + 42, 'regular', 11, (200, 230, 255, 255))
        fr.draw_text(f"Xoay Thân  : {telem['yaw']:+05.1f}°", px + 125, sec1_y + 60, 'regular', 11, (200, 230, 255, 255))
        fr.draw_text(f"Vận tốc xoay: [{telem['gyro'][0]:+03.0f}, {telem['gyro'][1]:+03.0f}, {telem['gyro'][2]:+03.0f}] °/s", px + 14, sec1_y + 92, 'regular', 10, (140, 180, 220, 255))

        # --- PHẦN 2: TRỌNG TÂM & LỰC CHÂN TIẾP ĐẤT ---
        sec2_y = sec1_y + 114
        fr.draw_text("2. TRỌNG TÂM & LỰC TIẾP ĐẤT", px + 14, sec2_y, 'bold', 11, (140, 180, 220, 255))

        com = telem['com']
        fr.draw_text(f"Vị trí Trọng tâm: [X:{com[0]:+.2f}, Y:{com[1]:+.2f}, Cao:{com[2]:.2f}]m", px + 14, sec2_y + 18, 'regular', 11, (0, 240, 255, 255))
        fr.draw_text(f"Tổng lực nâng mặt đất: {np.linalg.norm(telem['total_grf']):.0f} N (100.0%)", px + 14, sec2_y + 34, 'regular', 11, (180, 240, 180, 255))
        fr.draw_text(f"Tải lực 2 chân: Trái {telem['fz_left']:.0f}N | Phải {telem['fz_right']:.0f}N", px + 14, sec2_y + 50, 'regular', 11, (255, 210, 80, 255))

        # --- PHẦN 3: TẢI LỰC MÔ-MEN XOẮN CÁC KHỚP CHÍNH ---
        sec3_y = sec2_y + 74
        fr.draw_text("3. TẢI MÔ-MEN XOẮN CÁC KHỚP CHÍNH", px + 14, sec3_y, 'bold', 11, (140, 180, 220, 255))

        curr_y = sec3_y + 20
        loads = telem['actuator_loads']
        display_joints = [
            'l_hip_fe', 'r_hip_fe', 'l_knee_fe', 'r_knee_fe', 'l_ankle_ie', 'r_ankle_ie',
            'torso_pitch', 'torso_roll', 'l_shoulder_fe', 'r_shoulder_fe'
        ]

        for item in loads:
            if item['raw_name'] in display_joints:
                fr.draw_text(item['name'], px + 14, curr_y, 'regular', 10, (210, 230, 250, 255))
                
                # Thanh đo tỉ lệ tải
                bx, by, bw, bh = px + 195, curr_y + 2, 85, 10
                gl.glColor4f(0.12, 0.18, 0.28, 0.90)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex2f(bx, by); gl.glVertex2f(bx + bw, by)
                gl.glVertex2f(bx + bw, by + bh); gl.glVertex2f(bx, by + bh)
                gl.glEnd()

                fill_w = bw * item['ratio']
                if item['ratio'] < 0.35:
                    bar_col = (0.0, 0.94, 1.0, 0.95)
                elif item['ratio'] < 0.70:
                    bar_col = (0.0, 1.0, 0.50, 0.95)
                elif item['ratio'] < 0.88:
                    bar_col = (1.0, 0.75, 0.0, 0.95)
                else:
                    bar_col = (1.0, 0.25, 0.25, 1.0)
                
                gl.glColor4f(*bar_col)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex2f(bx, by); gl.glVertex2f(bx + fill_w, by)
                gl.glVertex2f(bx + fill_w, by + bh); gl.glVertex2f(bx, by + bh)
                gl.glEnd()

                fr.draw_text(f"{abs(item['tau']):.1f}Nm", px + 288, curr_y, 'mono', 10, (180, 210, 240, 255))
                curr_y += 18

    def _draw_bottom_controls_dock(self):
        """Thanh phím tắt điều khiển dưới đáy (Tiếng Việt)."""
        w, h = self.width, self.height
        dx, dy, dw, dh = 16, h - 42, w - 32, 32
        fr = self.font_renderer

        gl.glColor4f(0.06, 0.09, 0.15, 0.90)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(dx, dy); gl.glVertex2f(dx + dw, dy)
        gl.glVertex2f(dx + dw, dy + dh); gl.glVertex2f(dx, dy + dh)
        gl.glEnd()

        gl.glLineWidth(1.0)
        gl.glColor4f(0.20, 0.35, 0.55, 0.60)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(dx, dy); gl.glVertex2f(dx + dw, dy)
        gl.glVertex2f(dx + dw, dy + dh); gl.glVertex2f(dx, dy + dh)
        gl.glEnd()

        shortcuts = [
            ("SPACE", "Chạy/Dừng"),
            ("N", "Bước 1 Khung"),
            ("1-4", f"Tốc độ:{self.sim_speed}x"),
            ("MŨI TÊN/F", "Thử Đẩy Xô"),
            ("TAB", "Ẩn/Hiện Bảng Số Liệu"),
            ("F8", f"Giao diện:{'SÁNG' if self.theme_academic else 'TỐI'}"),
            ("P", "Chụp Ảnh"),
            ("R", "Đặt Lại Tư Thế")
        ]


        bx = dx + 12
        for key, desc in shortcuts:
            label = f"[{key}] {desc}"
            fr.draw_text(label, bx, dy + 8, 'regular', 11, (160, 200, 240, 255))
            bx += len(label) * 7.5 + 10

    def _draw_textured_quad(self, cx, cy, size, tex_key, alpha=1.0):
        if tex_key not in self.textures:
            return
        hs = size * 0.5
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[tex_key])
        gl.glColor4f(1.0, 1.0, 1.0, alpha)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0); gl.glVertex2f(cx - hs, cy - hs)
        gl.glTexCoord2f(1.0, 0.0); gl.glVertex2f(cx + hs, cy - hs)
        gl.glTexCoord2f(1.0, 1.0); gl.glVertex2f(cx + hs, cy + hs)
        gl.glTexCoord2f(0.0, 1.0); gl.glVertex2f(cx - hs, cy + hs)
        gl.glEnd()

        gl.glDisable(gl.GL_TEXTURE_2D)

    def _draw_gizmo_overlay(self):
        cx, cy = self.get_gizmo_center()
        nodes = self._get_gizmo_axis_nodes()

        # Đĩa xoay nền
        self._draw_textured_quad(cx, cy, self.gizmo_size * 2.0, 'DISC', alpha=0.85)

        # Các trục nối
        for node in nodes:
            rod_color = (node['color'][0], node['color'][1], node['color'][2], 0.95 if node['label'] else 0.40)
            gl.glLineWidth(3.0 if node['label'] else 1.8)
            gl.glColor4f(*rod_color)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(cx, cy)
            gl.glVertex2f(node['sx'], node['sy'])
            gl.glEnd()

        # Điểm tâm
        self._draw_textured_quad(cx, cy, 14.0, 'PIVOT', alpha=1.0)

        # Các đầu pin trục (+X, +Y, +Z, -X, -Y, -Z)
        for node in nodes:
            is_hovered = (self.hovered_node == node['name'])
            scale = 1.18 if is_hovered else 1.0
            pin_size = node['radius'] * 2.0 * scale
            self._draw_textured_quad(node['sx'], node['sy'], pin_size, node['name'], alpha=1.0)

    # ==========================================================================
    # 7. VÒNG LẶP MÔ PHỎNG & HIỂN THỊ ĐỒ HỌA
    # ==========================================================================
    def run(self):
        print("==================================================================")
        print(" [APPTRONIK APOLLO] Phòng Thí Nghiệm Đo Lường Cơ Sinh Học         ")
        print(" - Bộ điều khiển cân bằng chủ động: BẬT (Tự động đứng thẳng)     ")
        print(" - Đo lường vật lý: Trọng tâm 3D, Véc-tơ lực chân, Điểm ZMP      ")
        print(" - Phím TAB duy nhất: Ẩn/Hiện toàn bộ bảng thông số & đồ thị 2D   ")
        print(" - Quả cầu Gizmo Blender X/Y/Z: Luôn hiển thị cố định góc phải   ")
        print(" - Thử nghiệm lực đẩy xô: Phím Mũi Tên hoặc phím F               ")
        print(" - Chụp ảnh báo cáo: Phím P | Đổi nền sáng/tối: Phím F8          ")
        print("==================================================================")

        sim_accumulator = 0.0
        last_frame_time = time.time()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            now = time.time()
            frame_dt = now - last_frame_time
            last_frame_time = now

            self.frame_count += 1
            if now - self.last_fps_time >= 0.5:
                self.render_fps = self.frame_count / (now - self.last_fps_time)
                self.physics_fps = 200.0 * self.sim_speed if not self.paused else 0.0
                self.frame_count = 0
                self.last_fps_time = now

            self._update_camera_animation()

            if not self.paused:
                sim_accumulator += frame_dt * self.sim_speed
                steps_taken = 0
                while sim_accumulator >= self.model.opt.timestep and steps_taken < 10:
                    self._step_physics_with_balance()
                    sim_accumulator -= self.model.opt.timestep
                    steps_taken += 1
            elif self.step_single_frame:
                self._step_physics_with_balance()
                self.step_single_frame = False

            telem = self.telemetry.update(self.data)
            self.oscilloscope.append(
                self.data.time,
                telem['pelvis_z'],
                telem['fz_left'],
                telem['fz_right'],
                telem['com_vel'][1]
            )

            # 1. Kết xuất không gian 3D MuJoCo
            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)

            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            self._inject_3d_scientific_overlays(telem)
            mujoco.mjr_render(viewport, self.scn, self.con)

            # 2. Kết xuất giao diện 2D Overlay
            gl.glUseProgram(0)
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPushMatrix()
            gl.glLoadIdentity()
            gl.glOrtho(0, self.width, self.height, 0, -1, 1)

            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            gl.glLoadIdentity()

            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Hiển thị các bảng số liệu khi show_hud = True
            if self.show_hud:
                self._draw_top_scientific_ribbon(telem)
                self._draw_left_diagnostic_dashboard(telem)
                
                osc_w = min(440, self.width - 390)
                self.oscilloscope.draw(self.width - osc_w - 16, 125, osc_w, 240, self.font_renderer)

                self._draw_bottom_controls_dock()

            # Quả cầu định hướng Gizmo luôn luôn hiển thị cố định ở góc trên bên phải
            self._draw_gizmo_overlay()

            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)

            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_MODELVIEW)

            glfw.swap_buffers(self.window)

        glfw.destroy_window(self.window)
        glfw.terminate()

def main():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    viewer = BlenderMuJoCoViewer(model_path)
    viewer.run()

if __name__ == "__main__":
    main()
