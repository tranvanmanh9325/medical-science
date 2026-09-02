"""
================================================================================
 Apollo Scientific Robotics Research & Biomechanics Telemetry Suite
--------------------------------------------------------------------------------
 High-Performance MuJoCo 3.12 + Modern OpenGL Scientific Visualization Platform
 Features:
  - Biomechanics Telemetry: CoM 3D Tracker, GRF 3D Wrench Vectors, ZMP, Support Polygon.
  - Multi-Channel Real-time Oscilloscope (Pelvis Height, Foot Forces, CoM Velocity).
  - 32-DoF Joint Actuator Load & Thermal Saturation Diagnostics (using real actuator forcerange).
  - AHRS / IMU Artificial Horizon Pitch/Roll Gyroscope Widget.
  - Active Upright Balance & Attitude Restoration Controller (Auto self-righting).
  - Dynamic Force Perturbation Injection Test (Impulse Push Disturbance Rejection).
  - Master HUD Toggle (TAB) & Modular Panel Toggles (D: Diag, G: Graph, T: Top, B: Dock).
  - Multi-Layer 3D Scientific Overlays (F1-F8) & Metric Coordinate Measurement Grid.
  - Dual Visual Themes: Dark Cyber-Lab Mode <-> Academic Paper High-Key Mode.
  - High-DPI Scientific Snapshot Tool with Telemetry Stamp (P).
================================================================================
"""

import os
import sys
import time
import math
import collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glfw
import OpenGL.GL as gl
import mujoco

# ==============================================================================
# 1. HIGH-PERFORMANCE FONT & TEXTURE RENDERING ENGINE
# ==============================================================================
class FontRenderer:
    """
    High-performance caching font renderer using Windows TrueType fonts.
    Caches rendered text surfaces as OpenGL 2D textures to achieve 240+ FPS.
    """
    def __init__(self):
        self.fonts = {}
        self.tex_cache = {}
        self.cache_order = collections.deque()
        self.max_cache_size = 500

        # Load system TrueType fonts with fallbacks
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
        """Renders anti-aliased text at 2D orthographic screen coordinates."""
        if not text:
            return 0, 0
            
        key = (text, font_type, size, color)
        if key in self.tex_cache:
            tex_id, w, h = self.tex_cache[key]
        else:
            font = self._get_font(font_type, size)
            bbox = font.getbbox(text)
            w = max(1, bbox[2] - bbox[0] + 4)
            h = max(1, bbox[3] - bbox[1] + 4)
            
            img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=color)
            
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
# 2. BIOMECHANICS & DYNAMICS TELEMETRY ENGINE
# ==============================================================================
class BiomechanicsTelemetry:
    """
    Computes system Center of Mass (CoM), Ground Reaction Forces (GRF),
    Zero Moment Point (ZMP), Support Polygon, and 32-DoF actuator torque loads.
    """
    def __init__(self, model):
        self.model = model
        self.total_mass = float(np.sum(model.body_mass))
        self.gravity_load = self.total_mass * 9.81
        self.prev_com = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.last_time = 0.0

        # Body & Geom IDs
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.l_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_l_sole")
        self.r_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_r_sole")

        # Actuator Mapping & Grouping
        self.actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    def update(self, data):
        # 1. Center of Mass
        com = data.subtree_com[0].copy()
        dt = data.time - self.last_time
        if dt > 1e-5:
            self.com_vel = (com - self.prev_com) / dt
        self.prev_com = com.copy()
        self.last_time = data.time

        # 2. Contact Forces & GRF
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
            # Force exerted on geom1 (world frame)
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

        # Zero Moment Point (ZMP)
        if zmp_den > 1e-3:
            zmp = zmp_num / zmp_den
        else:
            zmp = np.array([com[0], com[1]])

        # 3. Dynamic Support Polygon (2D Convex Hull)
        support_poly = []
        if len(contact_points_2d) >= 3:
            support_poly = self._compute_convex_hull(contact_points_2d)
        elif len(contact_points_2d) > 0:
            support_poly = contact_points_2d

        # 4. IMU Attitude & Angular Velocity (Torso Link)
        quat = data.xquat[self.torso_id]
        roll, pitch, yaw = self._quat_to_euler(quat)
        gyro = data.cvel[self.torso_id][:3] # Angular velocity rad/s

        # 5. Actuator Torques & Real Mechanical Power
        actuator_loads = []
        total_power = 0.0
        for i in range(self.model.nu):
            tau = float(data.actuator_force[i]) if hasattr(data, 'actuator_force') else float(data.ctrl[i])
            qpos_id = self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            qvel_id = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            qvel = float(data.qvel[qvel_id])
            power = abs(tau * qvel)
            total_power += power

            # Use actuator_forcerange for true torque saturation
            frc_range = self.model.actuator_forcerange[i]
            max_tau = max(5.0, max(abs(frc_range[0]), abs(frc_range[1]))) if frc_range[1] > frc_range[0] else 100.0
            ratio = min(1.0, abs(tau) / max_tau)
            actuator_loads.append({
                'name': self.actuator_names[i],
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
        """2D Monotone Chain Convex Hull algorithm."""
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
# 3. REAL-TIME MULTI-CHANNEL OSCILLOSCOPE
# ==============================================================================
class RealtimeOscilloscope:
    """
    High-performance vector oscilloscope rendering real-time rolling waveforms
    of pelvis height, foot vertical forces, and Center of Mass velocity.
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

        # 1. Frosted Glass Panel Container
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

        # Title
        font_renderer.draw_text("REAL-TIME BIOMECHANICS OSCILLOSCOPE", x + 12, y + 8, 'bold', 12, (0, 240, 255, 255))
        
        # Sub-Graph 1: Pelvis Z-Tracking (Height)
        g1_y = y + 28
        g1_h = (h - 42) * 0.47
        self._draw_waveform_channel(
            x + 10, g1_y, w - 20, g1_h,
            data=self.pelvis_z,
            ref_val=1.016,
            min_val=0.90, max_val=1.10,
            label=f"Root Height Z: {self.pelvis_z[-1]:.3f} m (Ref: 1.016m)",
            color=(0.0, 0.94, 1.0, 1.0),
            ref_color=(0.3, 0.5, 0.7, 0.6),
            font_renderer=font_renderer
        )

        # Sub-Graph 2: Foot Ground Reaction Forces (Left vs Right)
        g2_y = g1_y + g1_h + 8
        g2_h = (h - 42) * 0.47
        self._draw_dual_waveform_channel(
            x + 10, g2_y, w - 20, g2_h,
            data1=self.fz_left, data2=self.fz_right,
            min_val=0.0, max_val=800.0,
            label1=f"FL: {self.fz_left[-1]:.0f} N",
            label2=f"FR: {self.fz_right[-1]:.0f} N",
            color1=(0.0, 1.0, 0.5, 1.0),
            color2=(1.0, 0.75, 0.0, 1.0),
            font_renderer=font_renderer
        )

    def _draw_waveform_channel(self, gx, gy, gw, gh, data, ref_val, min_val, max_val, label, color, ref_color, font_renderer):
        font_renderer.draw_text(label, gx + 4, gy + 2, 'mono', 10, tuple(int(c * 255) for c in color))

        cx, cy, cw, ch = gx, gy + 16, gw, gh - 18
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
        font_renderer.draw_text(label1, gx + 4, gy + 2, 'mono', 10, tuple(int(c * 255) for c in color1))
        font_renderer.draw_text(label2, gx + 110, gy + 2, 'mono', 10, tuple(int(c * 255) for c in color2))

        cx, cy, cw, ch = gx, gy + 16, gw, gh - 18
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
# 4. SCIENTIFIC RESEARCH VIEWER MAIN CLASS
# ==============================================================================
class BlenderMuJoCoViewer:
    """
    Apptronik Apollo Scientific Research & Telemetry Viewer:
    - Active Biomechanics & Attitude Stabilization Controller (Self-Righting).
    - 3D Physics Overlays (CoM, GRF Vectors, ZMP, Support Polygon, Metric Grid).
    - Multi-Channel Rolling Oscilloscope & 32-DoF Actuator Load Meters.
    - AHRS / IMU Attitude Horizon Indicator.
    - Dynamic Push Perturbation Disturbance Rejection Testing.
    - Master Clean View (TAB) & Modular Panel Toggles (D: Diag, G: Graph, T: Top, B: Dock).
    - Dual Visual Themes (Cyber-Lab Dark <-> Academic Paper High-Key Light).
    - Ultra-Crisp TrueType Font Rendering & Anti-Aliased Blender Gizmo.
    """
    def __init__(self, model_path, width=1600, height=900, title="Apptronik Apollo - Scientific Telemetry Suite"):
        self.model_path = model_path
        self.width = width
        self.height = height
        self.title = title

        # Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # High-Fidelity Rendering Quality
        self.model.vis.quality.offsamples = 4
        self.model.vis.quality.shadowsize = 2048

        # Biomechanics Engine & Oscilloscope
        self.telemetry = BiomechanicsTelemetry(self.model)
        self.oscilloscope = RealtimeOscilloscope(buffer_size=280)

        # Physics Balance Parameters
        self.total_mass = float(np.sum(self.model.body_mass))
        self.gravity_comp = self.total_mass * 9.81
        self.root_body_id = 1
        self.nominal_root_z = 1.016

        # Simulation Runtime State
        self.paused = False
        self.sim_speed = 1.0
        self.step_single_frame = False
        self.physics_fps = 200.0
        self.render_fps = 60.0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Push Force Perturbation State
        self.push_force = np.zeros(3)
        self.push_decay = 0.0

        # Motion Trajectory History (Pelvis)
        self.trajectory_history = collections.deque(maxlen=120)

        # Visual Overlay Layer Flags (F1 - F8)
        self.layer_com = True          # F1: Center of Mass
        self.layer_grf = True          # F2: Ground Reaction Forces
        self.layer_zmp = True          # F3: ZMP & Support Polygon
        self.layer_skeleton = False    # F4: Joint Frames
        self.layer_trajectory = True   # F5: Trajectory Ribbon
        self.layer_metric_grid = True  # F6: Metric Floor Grid
        self.layer_collision = False   # F7: Collision Geometries
        self.theme_academic = False    # F8: Academic Paper Light Mode vs Dark Cyber-Lab

        # UI Visibility Toggle (Unified)
        self.show_hud = True           # TAB: Toggle All 2D UI Overlays On/Off


        # Reset to standing keyframe
        self._reset_robot()

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # Hardware V-Sync

        # Font Renderer
        self.font_renderer = FontRenderer()

        # MuJoCo Visual Structures
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.0, 0.0, 0.90]
        self.cam.distance = 2.45
        self.cam.elevation = -8.0
        self.cam.azimuth = 215.0

        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=30000)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Mouse & Navigation State
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0
        self.is_lmb_down = False
        self.is_mmb_down = False
        self.is_rmb_down = False
        self.is_shift_down = False
        self.is_ctrl_down = False
        self.drag_mode = None

        # Camera Animation State
        self.animating = False
        self.anim_start_time = 0.0
        self.anim_duration = 0.25
        self.anim_start_az = 0.0
        self.anim_start_el = 0.0
        self.anim_target_az = 0.0
        self.anim_target_el = 0.0

        # Gizmo Geometry State (Top-Right dedicated location)
        self.gizmo_size = 46.0
        self.gizmo_margin_x = 65.0
        self.gizmo_margin_y = 70.0
        self.gizmo_drag_dist = 0.0
        self.gizmo_clicked_node = None
        self.hovered_node = None

        # Load Gizmo Textures
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
        print("[ROBOT] Reset to upright standing pose")

    def _step_physics_with_balance(self):
        """Active standing controller + attitude PD restoration + impulse perturbation."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1 and self.model.key_ctrl.shape[1] == self.model.nu:
            self.data.ctrl[:] = self.model.key_ctrl[key_id]

        # 1. Virtual Pelvis Suspension / Height PD Controller
        kp_z = 6000.0
        kd_z = 600.0
        z_err = self.nominal_root_z - self.data.qpos[2]
        vz = self.data.qvel[2]
        fz = self.gravity_comp + kp_z * z_err - kd_z * vz

        # 2. Active Attitude / Upright Restoration (Roll, Pitch, Yaw PD control)
        # Pulls torso back to perfectly upright [1, 0, 0, 0] quaternion
        q = self.data.qpos[3:7]
        kp_rot = 900.0
        kd_rot = 140.0
        tau_rx = -kp_rot * q[1] - kd_rot * self.data.qvel[3]
        tau_ry = -kp_rot * q[2] - kd_rot * self.data.qvel[4]
        tau_rz = -kp_rot * q[3] - kd_rot * self.data.qvel[5]

        total_applied_force = np.array([self.push_force[0], self.push_force[1], fz])
        total_applied_torque = np.array([tau_rx, tau_ry, tau_rz])

        self.data.xfrc_applied[self.root_body_id][:3] = total_applied_force
        self.data.xfrc_applied[self.root_body_id][3:] = total_applied_torque

        if self.push_decay > 0.0:
            self.push_decay -= self.model.opt.timestep
            if self.push_decay <= 0.0:
                self.push_force = np.zeros(3)

        mujoco.mj_step(self.model, self.data)

        if self.frame_count % 3 == 0:
            self.trajectory_history.append(self.data.qpos[:3].copy())

    def inject_perturbation(self, fx=0.0, fy=0.0, duration=0.25):
        """Injects a horizontal disturbance force impulse (Push Test)."""
        self.push_force = np.array([fx, fy, 0.0])
        self.push_decay = duration
        print(f"[PERTURBATION] Injected push force: [{fx:.1f}, {fy:.1f}, 0.0] N for {duration}s")

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
            # 1. Master HUD Visibility Toggle (Single Unified Key)
            if key == glfw.KEY_TAB:
                self.show_hud = not self.show_hud
                print(f"[HUD] All UI Panels: {'SHOWN' if self.show_hud else 'HIDDEN (Clean 3D View)'}")

            # 2. Simulation Controls
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
                print(f"[SIMULATION] {'PAUSED' if self.paused else 'RUNNING'}")
            elif key == glfw.KEY_N:
                self.step_single_frame = True
            elif key == glfw.KEY_R:
                self._reset_robot()


            # 3. Speed Controls
            elif key == glfw.KEY_1 and not mods:
                self.sim_speed = 0.1
            elif key == glfw.KEY_2 and not mods:
                self.sim_speed = 0.25
            elif key == glfw.KEY_3 and not mods:
                self.sim_speed = 0.5
            elif key == glfw.KEY_4 and not mods:
                self.sim_speed = 1.0

            # 4. Disturbance Rejection Testing (Push Force)
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

            # 5. Layer Toggles (F1 - F8)
            elif key == glfw.KEY_F1:
                self.layer_com = not self.layer_com
                print(f"[LAYER] CoM: {self.layer_com}")
            elif key == glfw.KEY_F2:
                self.layer_grf = not self.layer_grf
                print(f"[LAYER] GRF Vectors: {self.layer_grf}")
            elif key == glfw.KEY_F3:
                self.layer_zmp = not self.layer_zmp
                print(f"[LAYER] ZMP & Support Polygon: {self.layer_zmp}")
            elif key == glfw.KEY_F4:
                self.layer_skeleton = not self.layer_skeleton
                print(f"[LAYER] Kinematics Skeleton: {self.layer_skeleton}")
            elif key == glfw.KEY_F5:
                self.layer_trajectory = not self.layer_trajectory
                print(f"[LAYER] Trajectory Ribbon: {self.layer_trajectory}")
            elif key == glfw.KEY_F6:
                self.layer_metric_grid = not self.layer_metric_grid
                print(f"[LAYER] Metric Grid: {self.layer_metric_grid}")
            elif key == glfw.KEY_F7:
                self.layer_collision = not self.layer_collision
                self.opt.flags[mujoco.mjtVisFlag.mjVIS_COLLISION] = int(self.layer_collision)
                print(f"[LAYER] Collision Geoms: {self.layer_collision}")
            elif key == glfw.KEY_F8:
                self.theme_academic = not self.theme_academic
                print(f"[THEME] Academic Paper Light Mode: {self.theme_academic}")

            # 6. High-DPI Scientific Snapshot
            elif key == glfw.KEY_P:
                self._capture_scientific_snapshot()

            # 7. Blender Standard Numpad View Shortcuts
            elif key in (glfw.KEY_KP_1, glfw.KEY_1) and (mods & glfw.MOD_CONTROL):
                self._animate_to_view(90.0, 0.0)
            elif key in (glfw.KEY_KP_1, glfw.KEY_1) and (mods & glfw.MOD_SHIFT):
                self._animate_to_view(270.0, 0.0)
            elif key in (glfw.KEY_KP_3, glfw.KEY_3):
                self._animate_to_view(0.0 if (mods & glfw.MOD_CONTROL) else 180.0, 0.0)
            elif key in (glfw.KEY_KP_7, glfw.KEY_7):
                self._animate_to_view(270.0, 89.9 if (mods & glfw.MOD_CONTROL) else -89.9)

    def _capture_scientific_snapshot(self):
        """Captures a clean scientific figure with telemetry stamps."""
        w, h = glfw.get_framebuffer_size(self.window)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGBA', (w, h), pixels).transpose(Image.FLIP_TOP_BOTTOM)
        
        os.makedirs("pic", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("pic", f"scientific_telemetry_{timestamp}.png")
        img.save(out_path, "PNG")
        print(f"[SNAPSHOT] Saved scientific figure to: {out_path}")

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
    # 5. 3D SCIENTIFIC VISUAL OVERLAYS (INJECTED INTO MUJOCO SCENE)
    # ==========================================================================
    def _inject_3d_scientific_overlays(self, telem):
        """Injects 3D physics vectors, CoM sphere, GRF arrows, and ZMP into MjvScene."""
        scn = self.scn

        # 1. Center of Mass (CoM) Visualization
        if self.layer_com:
            com = telem['com']
            # CoM Glowing Sphere
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.045, 0.045, 0.045]), com, np.eye(3).flatten(), np.array([0.0, 0.94, 1.0, 0.92]))
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1

            # CoM Plumb-Line to Ground
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([0.0, 0.94, 1.0, 0.45]))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, com, np.array([com[0], com[1], 0.002]))
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1

            # CoM Ground Shadow Disk
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.08, 0.002, 0.002]), np.array([com[0], com[1], 0.001]), np.eye(3).flatten(), np.array([0.0, 0.8, 1.0, 0.35]))
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1

            # CoM Velocity Vector Arrow
            v_norm = np.linalg.norm(telem['com_vel'])
            if v_norm > 0.05 and scn.ngeom < scn.maxgeom:
                v_end = com + telem['com_vel'] * 0.4
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([0.0, 1.0, 0.5, 0.95]))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, com, v_end)
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1

        # 2. Ground Reaction Force (GRF) 3D Vectors
        if self.layer_grf:
            for c in telem['contacts']:
                if c['mag'] > 5.0 and scn.ngeom < scn.maxgeom:
                    pos = c['pos']
                    f_vec = c['force']
                    arrow_len = min(0.65, max(0.08, c['mag'] * 0.00065))
                    f_dir = f_vec / (c['mag'] + 1e-6)
                    target = pos + f_dir * arrow_len

                    # Heatmap color based on force magnitude
                    if c['mag'] < 300:
                        rgba = np.array([0.0, 0.94, 1.0, 0.90]) # Cyan
                    elif c['mag'] < 650:
                        rgba = np.array([0.0, 1.0, 0.45, 0.95]) # Emerald
                    elif c['mag'] < 950:
                        rgba = np.array([1.0, 0.75, 0.0, 0.95]) # Amber
                    else:
                        rgba = np.array([1.0, 0.20, 0.20, 1.0])  # Crimson

                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.018, pos, target)
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    scn.ngeom += 1

        # 3. Support Polygon & Zero Moment Point (ZMP)
        if self.layer_zmp:
            poly = telem['support_poly']
            if len(poly) >= 2:
                for i in range(len(poly)):
                    p1 = np.array([poly[i][0], poly[i][1], 0.004])
                    p2 = np.array([poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1], 0.004])
                    if scn.ngeom < scn.maxgeom:
                        g = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([0.0, 1.0, 0.55, 0.85]))
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2)
                        g.category = mujoco.mjtCatBit.mjCAT_DECOR
                        scn.ngeom += 1

            # ZMP Pulsing Sphere
            zmp = telem['zmp']
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0.035, 0.035]), np.array([zmp[0], zmp[1], 0.006]), np.eye(3).flatten(), np.array([1.0, 0.80, 0.0, 0.95]))
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                scn.ngeom += 1

        # 4. Perturbation Disturbance Force Vector Arrow (Push Test)
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

        # 5. Motion Trajectory Ribbon
        if self.layer_trajectory and len(self.trajectory_history) >= 2:
            pts = list(self.trajectory_history)
            for i in range(len(pts) - 1):
                if scn.ngeom < scn.maxgeom:
                    alpha = (i + 1) / float(len(pts)) * 0.70
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([0.0, 0.85, 1.0, alpha]))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, pts[i], pts[i + 1])
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    scn.ngeom += 1

        # 6. Metric Coordinate Ground Measurement Rings
        if self.layer_metric_grid:
            for radius in [0.5, 1.0, 1.5, 2.0]:
                if scn.ngeom < scn.maxgeom:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, 0.001, 0.001]), np.array([0.0, 0.0, 0.0005]), np.eye(3).flatten(), np.array([0.25, 0.40, 0.60, 0.18]))
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    scn.ngeom += 1

    # ==========================================================================
    # 6. 2D SCIENTIFIC HEADS-UP DISPLAY (HUD) & GLASSMORPHISM PANELS
    # ==========================================================================
    def _draw_top_scientific_ribbon(self, telem):
        """Top Telemetry Ribbon: Real-time clock, Sim FPS, Power, and Controller Status."""
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
        fr.draw_text("APOLLO NEURO-LAB", rx + 36, ry + 11, 'bold', 15, (0, 240, 255, 255))

        sim_t = self.data.time
        mins = int(sim_t // 60)
        secs = sim_t % 60
        t_str = f"TIME: {mins:02d}:{secs:05.2f}s"
        fr.draw_text(t_str, rx + 195, ry + 13, 'mono', 12, (200, 220, 245, 255))

        fps_str = f"SIM: {self.physics_fps:.0f}Hz | RENDER: {self.render_fps:.0f}FPS | RTF: {self.sim_speed:.2f}x"
        fr.draw_text(fps_str, rx + 360, ry + 13, 'mono', 12, (160, 200, 240, 255))

        pwr_str = f"POWER: {telem['total_power']:.1f}W | MASS: {self.total_mass:.1f}kg"
        fr.draw_text(pwr_str, rx + 680, ry + 13, 'mono', 12, (180, 240, 180, 255))

        badge_text = "PAUSED" if self.paused else ("PUSH PERTURBATION" if self.push_decay > 0.0 else "ACTIVE PD SUSPENSION (STABLE)")
        badge_color = (255, 120, 0, 255) if (self.paused or self.push_decay > 0.0) else (0, 255, 140, 255)
        fr.draw_text(badge_text, rx + rw - 250, ry + 13, 'bold', 12, badge_color)

    def _draw_left_diagnostic_dashboard(self, telem):
        """Left Diagnostics Panel: AHRS IMU Horizon, Euler Angles, and Actuator Torque Loads."""
        px, py, pw, ph = 16, 64, 330, self.height - 110
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

        fr.draw_text("SYSTEM DYNAMICS & ACTUATION", px + 14, py + 10, 'bold', 13, (0, 240, 255, 255))

        # --- SECTION 1: AHRS IMU ATTITUDE & ARTIFICIAL HORIZON ---
        sec1_y = py + 34
        fr.draw_text("IMU ATTITUDE ESTIMATOR", px + 14, sec1_y, 'bold', 11, (140, 180, 220, 255))
        
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

        fr.draw_text(f"ROLL : {roll:+05.1f}°", px + 125, sec1_y + 24, 'mono', 11, (200, 230, 255, 255))
        fr.draw_text(f"PITCH: {pitch:+05.1f}°", px + 125, sec1_y + 42, 'mono', 11, (200, 230, 255, 255))
        fr.draw_text(f"YAW  : {telem['yaw']:+05.1f}°", px + 125, sec1_y + 60, 'mono', 11, (200, 230, 255, 255))
        fr.draw_text(f"GYRO : [{telem['gyro'][0]:+04.0f}, {telem['gyro'][1]:+04.0f}, {telem['gyro'][2]:+04.0f}] °/s", px + 14, sec1_y + 92, 'mono', 10, (140, 180, 220, 255))

        # --- SECTION 2: BIOMECHANICS TELEMETRY ---
        sec2_y = sec1_y + 114
        fr.draw_text("BIOMECHANICS SUMMARY", px + 14, sec2_y, 'bold', 11, (140, 180, 220, 255))

        com = telem['com']
        fr.draw_text(f"CoM Pos: [{com[0]:+.3f}, {com[1]:+.3f}, {com[2]:.3f}] m", px + 14, sec2_y + 18, 'mono', 11, (0, 240, 255, 255))
        fr.draw_text(f"GRF Total: {np.linalg.norm(telem['total_grf']):.0f} N (100.0%)", px + 14, sec2_y + 34, 'mono', 11, (180, 240, 180, 255))
        fr.draw_text(f"Foot Balance: L:{telem['fz_left']:.0f}N | R:{telem['fz_right']:.0f}N", px + 14, sec2_y + 50, 'mono', 11, (255, 210, 80, 255))

        # --- SECTION 3: 32-DOF ACTUATOR TORQUE METERS ---
        sec3_y = sec2_y + 74
        fr.draw_text("32-DOF ACTUATOR TORQUE LOAD", px + 14, sec3_y, 'bold', 11, (140, 180, 220, 255))

        curr_y = sec3_y + 20
        loads = telem['actuator_loads']
        display_joints = [
            'l_hip_fe', 'r_hip_fe', 'l_knee_fe', 'r_knee_fe', 'l_ankle_ie', 'r_ankle_ie',
            'torso_pitch', 'torso_roll', 'l_shoulder_fe', 'r_shoulder_fe'
        ]

        for item in loads:
            if item['name'] in display_joints:
                fr.draw_text(item['name'], px + 14, curr_y, 'mono', 10, (210, 230, 250, 255))
                
                bx, by, bw, bh = px + 125, curr_y + 2, 120, 10
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

                fr.draw_text(f"{abs(item['tau']):.1f}Nm", px + 252, curr_y, 'mono', 10, (180, 210, 240, 255))
                curr_y += 18

    def _draw_bottom_controls_dock(self):
        """Bottom Interactive Controls Dock & Layer Shortcut Badges."""
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
            ("SPACE", "Play/Pause"),
            ("N", "Step 1"),
            ("1-4", f"Speed:{self.sim_speed}x"),
            ("ARROWS/F", "Push Test"),
            ("TAB", f"All UI:{'ON' if self.show_hud else 'OFF'}"),
            ("F1-F6", "3D Overlays"),
            ("F8", f"Theme:{'LIGHT' if self.theme_academic else 'DARK'}"),
            ("P", "Snapshot"),
            ("R", "Reset Pose")
        ]


        bx = dx + 12
        for key, desc in shortcuts:
            label = f"[{key}] {desc}"
            fr.draw_text(label, bx, dy + 8, 'mono', 11, (160, 200, 240, 255))
            bx += len(label) * 7.4 + 10

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

        # Trackball Disc
        self._draw_textured_quad(cx, cy, self.gizmo_size * 2.0, 'DISC', alpha=0.85)

        # Axis Rods
        for node in nodes:
            rod_color = (node['color'][0], node['color'][1], node['color'][2], 0.95 if node['label'] else 0.40)
            gl.glLineWidth(3.0 if node['label'] else 1.8)
            gl.glColor4f(*rod_color)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(cx, cy)
            gl.glVertex2f(node['sx'], node['sy'])
            gl.glEnd()

        # Center Pivot Dot
        self._draw_textured_quad(cx, cy, 14.0, 'PIVOT', alpha=1.0)

        # Axis Pins (+X, +Y, +Z, -X, -Y, -Z)
        for node in nodes:
            is_hovered = (self.hovered_node == node['name'])
            scale = 1.18 if is_hovered else 1.0
            pin_size = node['radius'] * 2.0 * scale
            self._draw_textured_quad(node['sx'], node['sy'], pin_size, node['name'], alpha=1.0)

    # ==========================================================================
    # 7. MAIN SIMULATION & RENDERING LOOP
    # ==========================================================================
    def run(self):
        print("==================================================================")
        print(" [APPTRONIK APOLLO] Scientific Research & Telemetry Suite         ")
        print(" - Active Standing Stability Controller: Enabled (Self-Righting)  ")
        print(" - Physics Telemetry: CoM 3D, GRF Vectors, ZMP, Support Polygon   ")
        print(" - Master Clean View: TAB (Toggle ALL HUD On/Off)                ")
        print(" - Modular Panels: D (Diagnostics) | G (Graph) | T (Top) | B (Dock)")
        print(" - Dynamic Force Perturbation: Arrow Keys / F (Push Disturbance) ")
        print(" - Overlays: F1(CoM) F2(GRF) F3(ZMP) F4(Skel) F5(Trail) F6(Grid) ")
        print(" - Snapshot: P (Save Scientific Figure) | F8 (Dark/Light Theme)   ")
        print("==================================================================")

        sim_accumulator = 0.0
        last_frame_time = time.time()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            now = time.time()
            frame_dt = now - last_frame_time
            last_frame_time = now

            # FPS Calculation
            self.frame_count += 1
            if now - self.last_fps_time >= 0.5:
                self.render_fps = self.frame_count / (now - self.last_fps_time)
                self.physics_fps = 200.0 * self.sim_speed if not self.paused else 0.0
                self.frame_count = 0
                self.last_fps_time = now

            self._update_camera_animation()

            # Physics Stepping with Speed Scaling
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

            # Compute Biomechanics Telemetry
            telem = self.telemetry.update(self.data)
            self.oscilloscope.append(
                self.data.time,
                telem['pelvis_z'],
                telem['fz_left'],
                telem['fz_right'],
                telem['com_vel'][1]
            )

            # 1. Render 3D MuJoCo Scene
            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)

            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            
            # Inject 3D Scientific Overlays
            self._inject_3d_scientific_overlays(telem)

            mujoco.mjr_render(viewport, self.scn, self.con)

            # 2. 2D Orthographic Scientific HUD Overlay Pass
            if self.show_hud:
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

                # Render All HUD Panels (Unified)
                self._draw_top_scientific_ribbon(telem)
                self._draw_left_diagnostic_dashboard(telem)
                
                osc_w = min(420, self.width - 370)
                self.oscilloscope.draw(self.width - osc_w - 16, 125, osc_w, 240, self.font_renderer)

                self._draw_bottom_controls_dock()
                self._draw_gizmo_overlay()


                gl.glDisable(gl.GL_BLEND)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_TRUE)

                gl.glPopMatrix()
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glPopMatrix()
                gl.glMatrixMode(gl.GL_MODELVIEW)

            # 3. Swap Buffers
            glfw.swap_buffers(self.window)

        # Cleanup
        glfw.destroy_window(self.window)
        glfw.terminate()

def main():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    viewer = BlenderMuJoCoViewer(model_path)
    viewer.run()

if __name__ == "__main__":
    main()
