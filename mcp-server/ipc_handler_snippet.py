# ═══════════════════════════════════════════════════════════════════
# MCP IPC HANDLER — Thêm vào BlenderMuJoCoViewer.__init__() và run loop
# Đọc lệnh từ mcp-server/ipc/sim_cmd.json (atomic file-based IPC)
# ═══════════════════════════════════════════════════════════════════

# --- Thêm vào __init__() sau khi load model: ---
import json as _json
self._ipc_cmd_file    = os.path.join(os.path.dirname(__file__), 'mcp-server', 'ipc', 'sim_cmd.json')
self._ipc_status_file = os.path.join(os.path.dirname(__file__), 'mcp-server', 'ipc', 'sim_status.json')
self._ipc_frame_count = 0
os.makedirs(os.path.dirname(self._ipc_status_file), exist_ok=True)

# --- Thêm method vào class (trước _step_physics_with_balance): ---
def _ipc_process_commands(self):
    '''Read and execute commands from MCP server (non-blocking, once per 10 frames).'''
    if self._ipc_frame_count % 10 != 0:
        self._ipc_frame_count += 1
        return
    self._ipc_frame_count += 1
    try:
        if not os.path.exists(self._ipc_cmd_file):
            return
        with open(self._ipc_cmd_file, 'r', encoding='utf-8') as f:
            cmd = _json.load(f)
        os.remove(self._ipc_cmd_file)  # Consume command
        t = cmd.get('type', '')
        if t == 'set_velocity':
            if self.policy and hasattr(self.policy, 'set_cmd_vel'):
                self.policy.set_cmd_vel(cmd.get('vx',0), cmd.get('vy',0), cmd.get('yaw',0))
            self._walk_vx = cmd.get('vx', 0); self._walk_vy = cmd.get('vy', 0); self._walk_yaw = cmd.get('yaw', 0)
        elif t == 'push':
            self.inject_perturbation(fx=cmd.get('fx',0), fy=cmd.get('fy',0), duration=cmd.get('duration',0.25))
        elif t == 'set_speed':
            self.sim_speed = float(cmd.get('value', 1.0))
        elif t == 'key':
            key_map = {'R': 'R', 'B': 'B', 'SPACE': 'SPACE', 'P': 'P'}
            k = cmd.get('key','')
            if k == 'R': self._reset_robot(); self.policy and self.policy.reset(); self.control_mode = 'PPO'
            elif k == 'B':
                if self.control_mode == 'RAGDOLL': self.control_mode = 'PPO'; self.policy and self.policy.reset()
                else: self.control_mode = 'RAGDOLL'
            elif k == 'SPACE': self.paused = not self.paused
            elif k == 'P': pass  # Screenshot handled separately
    except Exception: pass  # Non-blocking: ignore errors

def _ipc_write_status(self, telem):
    '''Write current robot status to IPC file for MCP server (non-blocking, once per 30 frames).'''
    if self._ipc_frame_count % 30 != 0:
        return
    try:
        qpos = self.data.qpos; qvel = self.data.qvel
        qw,qx,qy,qz = qpos[3],qpos[4],qpos[5],qpos[6]
        roll  = float(np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))))
        pitch = float(np.degrees(np.arcsin(max(-1,min(1,2*(qw*qy-qz*qx))))))
        yaw   = float(np.degrees(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))))
        status = {
            'control_mode': self.control_mode,
            'sim_time': float(self.data.time),
            'pelvis_z': float(qpos[2]),
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'vx_actual': float(qvel[0]), 'vy_actual': float(qvel[1]),
            'vx_cmd': float(self._walk_vx),
            'fz_left':  float(telem.get('fz_left', 0)),
            'fz_right': float(telem.get('fz_right', 0)),
            'total_power': float(telem.get('total_power', 0)),
            'total_mass': float(self.total_mass),
            'com_x': float(telem.get('com', [0,0,0])[0]) if isinstance(telem.get('com'), (list, np.ndarray)) else 0,
            'com_y': float(telem.get('com', [0,0,0])[1]) if isinstance(telem.get('com'), (list, np.ndarray)) else 0,
            'com_z': float(telem.get('com', [0,0,0])[2]) if isinstance(telem.get('com'), (list, np.ndarray)) else 0,
        }
        tmp = self._ipc_status_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            _json.dump(status, f)
        os.replace(tmp, self._ipc_status_file)
    except Exception: pass

# --- Thêm vào run loop (sau telem = self.telemetry.update(self.data)): ---
# self._ipc_process_commands()
# self._ipc_write_status(telem)
