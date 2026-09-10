"""
================================================================================
 Apollo Robot Simulation MCP Server  v1.0.0
 Model Context Protocol server ket noi voi Apollo MuJoCo simulation
================================================================================
 30+ tools chia 6 nhom:
   A. Simulation Control  (5 tools)
   B. Robot Control       (6 tools)
   C. Live Telemetry      (5 tools)
   D. Checkpoint & Model  (6 tools)
   E. Training Analysis   (6 tools)
   F. Advanced Tools      (4 tools)
================================================================================
"""
import asyncio, json, math, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

from mcp.server import MCPServer
from mcp.server.stdio import stdio_server

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RUN_BAT      = PROJECT_ROOT / "run.bat"
MAIN_PY      = PROJECT_ROOT / "main.py"
CK_DIR_S2    = PROJECT_ROOT / "colab_output" / "checkpoints_stage2"
TRAIN_LOG    = CK_DIR_S2 / "train.log"
IPC_DIR      = Path(__file__).parent / "ipc"
CMD_FILE     = IPC_DIR / "sim_cmd.json"
STATUS_FILE  = IPC_DIR / "sim_status.json"

mcp = MCPServer(
    name="apollo-robot-sim",
    title="Apollo Robot Simulation MCP Server",
    description="MCP server toan dien ket noi voi Apollo humanoid simulation (MuJoCo). 30+ tools.",
    version="1.0.0",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_sim_running() -> bool:
    try:
        r = subprocess.run(["wmic","process","where","name='python.exe'","get","CommandLine"],
                           capture_output=True, text=True, timeout=5)
        return "main.py" in r.stdout
    except Exception: return False

def _atomic_write(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

def _send_command(cmd: dict) -> dict:
    IPC_DIR.mkdir(exist_ok=True)
    cmd["ts"] = datetime.now().isoformat()
    _atomic_write(CMD_FILE, cmd)
    time.sleep(0.15)
    return {"success": True}

def _read_status() -> dict:
    try:
        if STATUS_FILE.exists() and time.time() - STATUS_FILE.stat().st_mtime < 5:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return {"error": "No live status. Simulation IPC not active."}
    except Exception as e: return {"error": str(e)}

def _find_ck(name: str) -> Path | None:
    for d in [CK_DIR_S2, PROJECT_ROOT / "kaggle_output"]:
        if not d.exists(): continue
        for f in d.rglob(name): return f
        direct = d / name
        if direct.exists(): return direct
    return None

def _load_ck_info(path: str) -> dict:
    try:
        ck = np.load(path, allow_pickle=True)
        keys = list(ck.keys())
        info = {
            "path": path, "filename": Path(path).name,
            "step": int(ck["_step"]) if "_step" in keys else None,
            "iter": int(ck["_it"])   if "_it"   in keys else None,
            "num_params": sum(ck[k].size for k in keys if not k.startswith("_")),
        }
        std_k = next((k for k in keys if "log_std" in k.lower()), None)
        if std_k:
            ls = ck[std_k]
            actual = -4.0 + 3.0 / (1.0 + np.exp(-ls))
            info["log_std_actual_mean"] = float(np.mean(actual))
            info["std_mean"] = float(np.mean(np.exp(actual)))
        nan_count = sum(1 for k in keys if not k.startswith("_") and
                        (np.any(np.isnan(ck[k])) or np.any(np.isinf(ck[k]))))
        info["healthy"] = nan_count == 0
        info["nan_count"] = nan_count
        return info
    except Exception as e: return {"error": str(e), "path": path}

import re as _re
_LOG_PAT = _re.compile(
    r"\[(\d+)/(\d+)\] steps=([\d,]+) \| rew=([\d.]+) \| loss=([\d.]+) "
    r"\| sps=([\d,]+) \| vx_max=([\d.]+) \| ent=([\d.]+) \| log_std=(-?[\d.]+) \| t=(\d+)s"
)
def _parse_log() -> list:
    if not TRAIN_LOG.exists(): return []
    entries = []
    with open(TRAIN_LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOG_PAT.search(line)
            if m:
                entries.append({
                    "iter": int(m.group(1)), "total": int(m.group(2)),
                    "steps": int(m.group(3).replace(",","")), "reward": float(m.group(4)),
                    "loss": float(m.group(5)), "sps": int(m.group(6).replace(",","")),
                    "vx_max": float(m.group(7)), "ent": float(m.group(8)),
                    "log_std": float(m.group(9)), "t": int(m.group(10)),
                    "status": "WALKING WELL" if "WALKING WELL" in line else
                              ("WALKING" if "WALKING" in line else "OTHER"),
                })
    return entries

# ══════════════════════════════════════════════════════════════════════════════
# A. SIMULATION CONTROL
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Khoi chay simulation", description="Launch Apollo 3D MuJoCo simulation via run.bat. Takes 3-5s to load.")
async def simulation_start() -> str:
    if _is_sim_running(): return "Simulation da chay roi!"
    subprocess.Popen(["cmd.exe","/c",str(RUN_BAT)], cwd=str(PROJECT_ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)
    await asyncio.sleep(1)
    return "Da khoi chay Apollo simulation! Cho 3-5s de cua so 3D xuat hien.\nPhim: W/S=tien/lui | UP/DN=toc do | R=reset | B=ragdoll | X=dung"

@mcp.tool(title="Dung simulation", description="Kill Apollo simulation process.")
async def simulation_stop() -> str:
    r = subprocess.run(["taskkill","/F","/IM","python.exe","/T"], capture_output=True, text=True, timeout=10)
    return "Da dung simulation." if r.returncode == 0 else f"Khong co process: {r.stderr.strip()}"

@mcp.tool(title="Trang thai simulation", description="Check if simulation is running and read live status from IPC.")
async def simulation_status() -> str:
    running = _is_sim_running()
    st = _read_status()
    lines = [f"Simulation: {'DANG CHAY' if running else 'KHONG CHAY'}"]
    if running and "error" not in st:
        lines += [
            f"Mode: {st.get('control_mode','?')}",
            f"Time: {st.get('sim_time',0):.2f}s",
            f"Pelvis Z: {st.get('pelvis_z',0):.3f}m",
            f"Vx: {st.get('vx_actual',0):.2f}m/s ({st.get('vx_actual',0)*3.6:.1f}km/h)",
            f"Roll/Pitch: {st.get('roll',0):.1f}/{st.get('pitch',0):.1f} deg",
        ]
    elif not running: lines.append("Dung 'simulation_start' de khoi dong.")
    else: lines.append(f"IPC chua ket noi: {st.get('error','')}")
    return "\n".join(lines)

@mcp.tool(title="Tam dung / Tiep tuc", description="Toggle pause/resume simulation (Space key equivalent).")
async def simulation_toggle_pause() -> str:
    _send_command({"type":"key","key":"SPACE"})
    return "Da gui lenh pause/resume."

@mcp.tool(title="Dat toc do mo phong", description="Set simulation speed. 0.1=slow 10x, 1.0=realtime, 2.0=fast 2x.")
async def simulation_set_speed(speed: float = 1.0) -> str:
    """Args: speed: Simulation speed factor (0.1 to 4.0)"""
    speed = max(0.05, min(4.0, speed))
    _send_command({"type":"set_speed","value":speed})
    return f"Da dat toc do: {speed}x"

# ══════════════════════════════════════════════════════════════════════════════
# B. ROBOT CONTROL
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Dat lenh van toc robot", description="Send velocity command to robot. vx: forward m/s [-1.2,1.2]. vy: lateral m/s. yaw: rotation rad/s.")
async def robot_set_velocity(vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0) -> str:
    """Args: vx: forward speed m/s [-1.2,1.2]. vy: lateral m/s [-0.3,0.3]. yaw: rotation rad/s [-0.5,0.5]."""
    vx=max(-1.2,min(1.2,vx)); vy=max(-0.3,min(0.3,vy)); yaw=max(-0.5,min(0.5,yaw))
    _send_command({"type":"set_velocity","vx":vx,"vy":vy,"yaw":yaw})
    return f"Lenh: vx={vx:+.2f}m/s ({abs(vx)*3.6:.1f}km/h), vy={vy:+.2f}, yaw={yaw:+.2f}rad/s"

@mcp.tool(title="Dung robot", description="Stop robot immediately (all velocity to 0). Equivalent to X key.")
async def robot_stop() -> str:
    _send_command({"type":"set_velocity","vx":0.0,"vy":0.0,"yaw":0.0})
    return "Robot da dung! cmd_vel=[0,0,0]"

@mcp.tool(title="Reset robot", description="Reset robot to default standing pose and restart FSM warmup. R key.")
async def robot_reset() -> str:
    _send_command({"type":"key","key":"R"})
    return "Robot da reset!\nFSM: Phase1(PD HOLD 1s) -> Phase2(Cosine Blend 1.5s) -> Phase3(Full PPO)"

@mcp.tool(title="Dat che do dieu khien", description="Switch control mode: 'ppo' (AI self-balance) or 'ragdoll' (motor off).")
async def robot_set_mode(mode: str = "ppo") -> str:
    """Args: mode: 'ppo' for AI control or 'ragdoll' for physics-only."""
    _send_command({"type":"key","key":"B"})
    return f"Da gui lenh chuyen sang che do: {mode}"

@mcp.tool(title="Ap dung luc day", description="Apply external push force to test balance recovery. fx=forward N, fy=lateral N, duration=seconds.")
async def robot_apply_push(fx: float = 150.0, fy: float = 0.0, duration: float = 0.25) -> str:
    """Args: fx: Forward force N. fy: Lateral force N. duration: Time in seconds."""
    _send_command({"type":"push","fx":fx,"fy":fy,"duration":duration})
    mag = math.sqrt(fx**2+fy**2)
    return f"Ap dung luc: [{fx:.0f},{fy:.0f}]N ({mag:.0f}N) trong {duration}s"

@mcp.tool(title="Kiem tra phuc hoi cu day", description="Run automated push recovery test sequence from multiple directions.")
async def robot_run_recovery_test() -> str:
    if not _is_sim_running(): return "Simulation chua chay."
    tests = [
        ("Day nhe truoc", 100,0,0.2), ("Day nhe sau",-100,0,0.2),
        ("Day trai",0,100,0.2), ("Day phai",0,-100,0.2),
        ("Day vua truoc",200,0,0.25), ("Day manh truoc",350,0,0.3),
        ("Day cheo",150,150,0.25),
    ]
    results = ["## Ket Qua Kiem Tra Phuc Hoi"]
    for name,fx,fy,dur in tests:
        _send_command({"type":"push","fx":fx,"fy":fy,"duration":dur})
        await asyncio.sleep(dur + 1.2)
        st = _read_status()
        z = st.get("pelvis_z",0); r = abs(st.get("roll",180))
        ok = z > 0.7 and r < 45
        results.append(f"{'OK' if ok else 'FAIL'} {name}: z={z:.3f}m roll={r:.1f}deg [{fx},{fy}]N")
    return "\n".join(results)

# ══════════════════════════════════════════════════════════════════════════════
# C. LIVE TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Trang thai robot hien tai", description="Get current robot state: position, velocity, orientation, control mode.")
async def robot_get_state() -> str:
    st = _read_status()
    if "error" in st:
        return f"IPC chua ket noi: {st['error']}\nCan them IPC handler vao main.py"
    return (
        f"Mode: {st.get('control_mode','?')} | Time: {st.get('sim_time',0):.3f}s\n"
        f"Pelvis Z: {st.get('pelvis_z',0):.4f}m\n"
        f"Roll: {st.get('roll',0):.2f}deg | Pitch: {st.get('pitch',0):.2f}deg\n"
        f"Vx: {st.get('vx_actual',0):.3f}m/s ({st.get('vx_actual',0)*3.6:.2f}km/h)\n"
        f"GRF L: {st.get('fz_left',0):.1f}N | R: {st.get('fz_right',0):.1f}N\n"
        f"Power: {st.get('total_power',0):.1f}W"
    )

@mcp.tool(title="Du lieu co sinh hoc", description="Get biomechanics data: CoM, ZMP, GRF, joint torques.")
async def robot_get_biomechanics() -> str:
    st = _read_status()
    if "error" in st: return f"IPC error: {st['error']}"
    return (
        f"CoM: ({st.get('com_x',0):.3f}, {st.get('com_y',0):.3f}, {st.get('com_z',0):.3f}) m\n"
        f"ZMP: ({st.get('zmp_x',0):.3f}, {st.get('zmp_y',0):.3f}) m\n"
        f"GRF Trai: {st.get('fz_left',0):.1f}N | Phai: {st.get('fz_right',0):.1f}N\n"
        f"Tong GRF: {st.get('fz_left',0)+st.get('fz_right',0):.1f}N"
    )

@mcp.tool(title="Du lieu IMU", description="Get IMU sensor data: Roll/Pitch/Yaw angles and stability assessment.")
async def robot_get_imu() -> str:
    st = _read_status()
    if "error" in st: return f"IPC error: {st['error']}"
    r,p,y = st.get("roll",0), st.get("pitch",0), st.get("yaw",0)
    stable = abs(r)<5 and abs(p)<10
    return (
        f"Roll:  {r:+.2f}deg {'OK' if abs(r)<5 else 'WARN' if abs(r)<15 else 'DANGER'}\n"
        f"Pitch: {p:+.2f}deg {'OK' if abs(p)<10 else 'WARN' if abs(p)<20 else 'DANGER'}\n"
        f"Yaw:   {y:+.2f}deg\n"
        f"Trang thai: {'On dinh' if stable else 'Mat can bang'}"
    )

@mcp.tool(title="Phan tich dang di", description="Analyze robot gait: speed tracking error, foot contact phase, step frequency.")
async def robot_get_gait_analysis() -> str:
    st = _read_status()
    if "error" in st: return f"IPC error: {st['error']}"
    vx_cmd = st.get("vx_cmd",0); vx_act = st.get("vx_actual",0)
    err = abs(vx_cmd-vx_act)
    fzl = st.get("fz_left",0); fzr = st.get("fz_right",0)
    total = fzl+fzr
    phase = "Chan Trai Nhac" if fzl<50 else ("Chan Phai Nhac" if fzr<50 else "Hai Chan Tru")
    return (
        f"Toc do lenh: {vx_cmd:.2f}m/s ({vx_cmd*3.6:.1f}km/h)\n"
        f"Toc do thuc: {vx_act:.2f}m/s ({vx_act*3.6:.1f}km/h)\n"
        f"Sai so: {err:.3f}m/s {'OK' if err<0.1 else 'WARN'}\n"
        f"GRF: T={fzl:.0f}N P={fzr:.0f}N\n"
        f"Pha buoc: {phase}"
    )

@mcp.tool(title="Cong suat tieu thu", description="Get power consumption data: mechanical power per joint and total.")
async def robot_get_power() -> str:
    st = _read_status()
    if "error" in st: return f"IPC error: {st['error']}"
    return (
        f"Tong cong suat co hoc: {st.get('total_power',0):.1f}W\n"
        f"Khoi luong robot: {st.get('total_mass',73):.1f}kg\n"
        f"Cost of Transport (CoT): {st.get('total_power',0)/(st.get('total_mass',73)*9.81*max(abs(st.get('vx_actual',0.01)),0.01)):.3f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# D. CHECKPOINT & MODEL
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Liet ke checkpoint", description="List all .npz checkpoint files with step and size info.")
async def checkpoint_list() -> str:
    lines = ["Danh Sach Checkpoint:"]
    for d in [CK_DIR_S2, PROJECT_ROOT/"kaggle_output"]:
        if not d.exists(): continue
        files = sorted(d.rglob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files: continue
        lines.append(f"\n{d.name}/")
        for f in files[:10]:
            try:
                ck = np.load(str(f)); step = int(ck["_step"]) if "_step" in ck else 0
                lines.append(f"  {f.name} ({f.stat().st_size//1024}KB) step={step:,}")
            except: lines.append(f"  {f.name} (unreadable)")
    return "\n".join(lines)

@mcp.tool(title="Thong tin checkpoint", description="Get detailed info about a specific checkpoint file.")
async def checkpoint_info(checkpoint_name: str = "apollo_stage2_v8_final.npz") -> str:
    """Args: checkpoint_name: Filename (e.g. apollo_stage2_v8_final.npz)"""
    p = _find_ck(checkpoint_name)
    if not p: return f"Khong tim thay: {checkpoint_name}. Dung 'checkpoint_list' xem danh sach."
    info = _load_ck_info(str(p))
    if "error" in info: return f"Loi: {info['error']}"
    return (
        f"File: {info['filename']}\n"
        f"Step: {info.get('step','N/A'):,}\n"
        f"Iter: {info.get('iter','N/A'):,}\n"
        f"Params: {info.get('num_params',0):,}\n"
        f"Health: {'SACH' if info.get('healthy') else 'CO NaN/Inf'}\n"
        f"log_std actual: {info.get('log_std_actual_mean','N/A'):.4f}\n"
        f"std (sigma): {info.get('std_mean','N/A'):.4f}"
    )

@mcp.tool(title="Kiem tra weights", description="Inspect network layer weights: norm, mean, std per layer.")
async def checkpoint_inspect_weights(checkpoint_name: str = "apollo_stage2_v8_final.npz") -> str:
    """Args: checkpoint_name: Checkpoint filename."""
    p = _find_ck(checkpoint_name)
    if not p: return f"Khong tim thay: {checkpoint_name}"
    ck = np.load(str(p))
    keys = [k for k in ck.keys() if not k.startswith("_")]
    lines = [f"Weights: {p.name}\n{'Layer':<45} {'Shape':<18} {'Norm':>8} {'Mean':>8} {'Std':>8}"]
    for k in sorted(keys):
        v = ck[k]
        lines.append(f"{k:<45} {str(v.shape):<18} {np.linalg.norm(v):>8.3f} {np.mean(v):>8.4f} {np.std(v):>8.4f}")
    return "\n".join(lines)

@mcp.tool(title="So sanh 2 checkpoint", description="Compare two checkpoints: steps, health, log_std.")
async def checkpoint_compare(ck_a: str = "apollo_stage2_v8_final.npz", ck_b: str = "apollo_stage2_v6_final.npz") -> str:
    """Args: ck_a: First checkpoint filename. ck_b: Second checkpoint filename."""
    pa = _find_ck(ck_a); pb = _find_ck(ck_b)
    ia = _load_ck_info(str(pa)) if pa else {"error": "Not found"}
    ib = _load_ck_info(str(pb)) if pb else {"error": "Not found"}
    return (
        f"So sanh Checkpoint:\n"
        f"{'Tham so':<20} {ck_a[:25]:<27} {ck_b[:25]}\n"
        f"{'Step':<20} {str(ia.get('step','?')):>25} {str(ib.get('step','?'))}\n"
        f"{'Iter':<20} {str(ia.get('iter','?')):>25} {str(ib.get('iter','?'))}\n"
        f"{'Params':<20} {str(ia.get('num_params','?')):>25} {str(ib.get('num_params','?'))}\n"
        f"{'Healthy':<20} {('SACH' if ia.get('healthy') else 'NaN'):>25} {('SACH' if ib.get('healthy') else 'NaN')}\n"
        f"{'log_std':<20} {str(round(ia.get('log_std_actual_mean',0),4)):>25} {str(round(ib.get('log_std_actual_mean',0),4))}"
    )

@mcp.tool(title="Timeline checkpoint", description="List checkpoints ordered by training step.")
async def checkpoint_timeline() -> str:
    if not CK_DIR_S2.exists(): return f"Khong tim thay: {CK_DIR_S2}"
    data = []
    for f in CK_DIR_S2.glob("*.npz"):
        try:
            ck = np.load(str(f))
            step = int(ck["_step"]) if "_step" in ck else 0
            data.append((f.name, step, f.stat().st_size//1024))
        except: pass
    data.sort(key=lambda x: x[1])
    lines = ["Timeline Checkpoint (step thu tu):"]
    for name,step,sz in data:
        pct = step/300_000_000*100
        bar = "=" * int(pct/5) + ">" + " " * (20-int(pct/5))
        lines.append(f"[{bar}] {pct:4.0f}% {step:>12,}  {name} ({sz}KB)")
    return "\n".join(lines)

@mcp.tool(title="Xuat info checkpoint", description="Export checkpoint info to JSON file in ipc/ directory.")
async def checkpoint_export_json(checkpoint_name: str = "apollo_stage2_v8_final.npz") -> str:
    """Args: checkpoint_name: Checkpoint filename to export."""
    p = _find_ck(checkpoint_name)
    if not p: return f"Khong tim thay: {checkpoint_name}"
    info = _load_ck_info(str(p))
    out = IPC_DIR / f"{checkpoint_name}.info.json"
    IPC_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return f"Da xuat: {out}\nNoi dung:\n{json.dumps(info, indent=2, default=str)[:800]}"

# ══════════════════════════════════════════════════════════════════════════════
# E. TRAINING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Doc log training", description="Read last N lines from train.log.")
async def training_log_read(last_n: int = 30) -> str:
    """Args: last_n: Number of recent lines to show."""
    if not TRAIN_LOG.exists(): return f"Khong tim thay: {TRAIN_LOG}"
    lines = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"Train Log ({len(lines)} dong tong, {last_n} cuoi):\n\n" + "\n".join(lines[-last_n:])

@mcp.tool(title="Tom tat chi so training", description="Parse train.log and show reward stats, curriculum, walking quality.")
async def training_metrics_summary() -> str:
    entries = _parse_log()
    if not entries: return f"Khong doc duoc train.log tai {TRAIN_LOG}"
    last = entries[-1]
    rewards = [e["reward"] for e in entries]
    ww = sum(1 for e in entries if e["status"]=="WALKING WELL")
    return (
        f"Tom Tat Training v8:\n"
        f"Iters: {entries[0]['iter']} -> {last['iter']} / {last['total']}\n"
        f"Steps: {last['steps']:,} ({last['steps']/1e6:.1f}M)\n"
        f"Thoi gian: {last['t']//3600}h {(last['t']%3600)//60}m\n"
        f"Reward: min={min(rewards):.5f} max={max(rewards):.5f} avg={sum(rewards)/len(rewards):.5f}\n"
        f"Reward cuoi: {last['reward']:.5f}\n"
        f"log_std cuoi: {last['log_std']:.4f} (std={math.exp(last['log_std']):.4f})\n"
        f"vx_max cuoi: {last['vx_max']}m/s ({last['vx_max']*3.6:.1f}km/h)\n"
        f"WALKING WELL: {ww}/{len(entries)} ({ww/len(entries)*100:.1f}%)\n"
        f"Trang thai: {'HOAN THANH' if last['iter']>=last['total'] else 'DANG CHAY'}"
    )

@mcp.tool(title="Tien do training", description="Show training progress with ASCII progress bar.")
async def training_progress() -> str:
    entries = _parse_log()
    if not entries: return "Khong co du lieu training"
    last = entries[-1]
    pct = last["iter"]/last["total"]*100
    bar = "=" * int(pct/4) + ">" + "." * (25-int(pct/4))
    return (
        f"Tien Do Training v8:\n[{bar}] {pct:.1f}%\n"
        f"Iter: {last['iter']:,}/{last['total']:,} | Steps: {last['steps']/1e6:.1f}M/300M\n"
        f"Reward: {last['reward']:.5f} | log_std: {last['log_std']:.4f} | vx_max: {last['vx_max']}m/s\n"
        f"Training time: {last['t']//3600}h {(last['t']%3600)//60}m\n"
        f"{'HOAN THANH!' if last['iter']>=last['total'] else 'Dang chay...'}"
    )

@mcp.tool(title="Phan tich curriculum vx_max", description="Analyze vx_max curriculum progression and reward at each speed level.")
async def training_vx_curriculum() -> str:
    entries = _parse_log()
    if not entries: return "Khong co du lieu"
    groups = {}
    for e in entries:
        vx = round(e["vx_max"],2)
        groups.setdefault(vx,[]).append(e["reward"])
    lines = [f"{'vx_max':>8} {'km/h':>6} {'Count':>6} {'Avg Reward':>12} {'Max':>10}"]
    lines.append("-"*48)
    for vx in sorted(groups):
        r = groups[vx]
        lines.append(f"{vx:>8.2f} {vx*3.6:>6.1f} {len(r):>6} {sum(r)/len(r):>12.5f} {max(r):>10.5f}")
    return "\n".join(lines)

@mcp.tool(title="Ve duong reward (ASCII)", description="Plot ASCII reward curve over training iterations.")
async def training_reward_curve(n_points: int = 25) -> str:
    """Args: n_points: Number of points to sample."""
    entries = _parse_log()
    if not entries: return "Khong co du lieu"
    step = max(1,len(entries)//n_points)
    sampled = entries[::step]
    if sampled[-1] != entries[-1]: sampled.append(entries[-1])
    min_r,max_r = min(e["reward"] for e in sampled), max(e["reward"] for e in sampled)
    rng = max_r-min_r if max_r>min_r else 0.01
    H = 8; rows = [[" "]*len(sampled) for _ in range(H)]
    for i,e in enumerate(sampled):
        row = H-1-int((e["reward"]-min_r)/rng*(H-1))
        rows[max(0,min(H-1,row))][i] = "|"
    lines = [f"Reward Curve [{min_r:.5f} .. {max_r:.5f}]"]
    for row in rows: lines.append("|"+"".join(row)+"|")
    lines.append("+" + "-"*len(sampled) + "+")
    return "\n".join(lines)

@mcp.tool(title="Tim kiem trong log", description="Search for pattern in training log (e.g. 'RESUME', 'ERROR', 'WALKING').")
async def training_log_search(pattern: str = "RESUME") -> str:
    """Args: pattern: Text to search (case-insensitive)."""
    if not TRAIN_LOG.exists(): return f"Khong tim thay: {TRAIN_LOG}"
    matches = []
    with open(TRAIN_LOG, encoding="utf-8", errors="replace") as f:
        for i,line in enumerate(f,1):
            if pattern.lower() in line.lower():
                matches.append(f"L{i:4}: {line.rstrip()}")
    if not matches: return f"Khong tim thay '{pattern}'"
    return f"Tim thay '{pattern}' ({len(matches)} ket qua):\n" + "\n".join(matches[:40])

# ══════════════════════════════════════════════════════════════════════════════
# F. ADVANCED TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool(title="Debug observation space", description="Inspect observation space structure and dimension of the policy.")
async def policy_obs_debug(checkpoint_name: str = "apollo_stage2_v8_final.npz") -> str:
    """Args: checkpoint_name: Checkpoint to inspect."""
    components = [
        ("upvec",3,"Body z-axis in world frame"),
        ("linvel",3,"Linear velocity world-frame m/s"),
        ("angvel",3,"Angular velocity rad/s"),
        ("jpos",32,"Joint pos relative to default rad"),
        ("jvel",32,"Joint velocities rad/s"),
        ("prev_act",32,"Previous action (raw)"),
        ("cmd_vel",3,"Velocity command [vx,vy,yaw]"),
        ("gait_phase",4,"CPG phase [sinL,cosL,sinR,cosR]"),
        ("foot_contact",2,"Foot contact [L,R] bool"),
    ]
    total = sum(d for _,d,_ in components)
    lines = [f"Obs Space ({total}D):\n{'Name':<14}{'Idx':>10}  Description"]
    idx = 0
    for name,d,desc in components:
        lines.append(f"{name:<14}[{idx:>3}:{idx+d:<3}]  {desc}")
        idx += d
    lines.append(f"\nTotal: {total}D {'= 114 OK' if total==114 else '!= 114 WARN'}")
    return "\n".join(lines)

@mcp.tool(title="Kiem tra sim-to-real gap", description="Check if main.py has all critical sim-to-real fixes applied.")
async def policy_sim2real_check() -> str:
    try: src = MAIN_PY.read_text(encoding="utf-8", errors="replace")
    except: return "Khong doc duoc main.py"
    checks = [
        ("SIM_DT     = 0.002",   "Timestep 0.002s (training)"),
        ("N_SUBSTEPS = 5",        "5 substeps = 100Hz policy"),
        ("IMPLICITFAST",          "Integrator IMPLICITFAST"),
        ("solref[i, 0] = 0.004", "Contact solref=0.004"),
        ("apollo_stage2_v8_final","v8 checkpoint priority"),
        ("PPOPolicyStage2V8",     "v8 LINEAR actor class"),
        ("STAND_HOLD_STEPS",      "3-phase FSM warmup"),
        ("cmd_magnitude",         "CPG freeze when standing"),
        ("MAX_ACT_RATE",          "Slew-rate limiter"),
    ]
    ok = []; fail = []
    for pattern,label in checks:
        (ok if pattern in src else fail).append(label)
    result = ["Sim-to-Real Gap Check:"]
    for f in fail: result.append(f"FAIL {f}")
    for o in ok:  result.append(f"OK   {o}")
    result.append(f"\n{'TẤT CẢ OK!' if not fail else f'CO {len(fail)} VAN DE!'}")
    return "\n".join(result)

@mcp.tool(title="Chup man hinh", description="Send screenshot command to running simulation (P key).")
async def simulation_screenshot() -> str:
    if not _is_sim_running(): return "Simulation chua chay."
    _send_command({"type":"key","key":"P"})
    return "Da gui lenh chup man hinh (phim P)."

@mcp.tool(title="Thong tin du an", description="Show complete project overview: training status, checkpoints, simulation.")
async def project_info() -> str:
    ck_p = CK_DIR_S2 / "apollo_stage2_v8_final.npz"
    ck = _load_ck_info(str(ck_p)) if ck_p.exists() else {}
    entries = _parse_log()
    last = entries[-1] if entries else {}
    return (
        f"Apollo Humanoid Robot Simulation - Du An\n"
        f"{'='*50}\n"
        f"MODEL: Stage 2 v8 (Walking AI)\n"
        f"  Step: {ck.get('step','?'):,}\n"
        f"  Health: {'SACH' if ck.get('healthy') else 'NaN'}\n"
        f"  log_std: {ck.get('log_std_actual_mean','?'):.4f}\n"
        f"\nTRAINING:\n"
        f"  Iters: {last.get('iter',0):,}/{last.get('total',3051):,}\n"
        f"  Reward: {last.get('reward',0):.5f}\n"
        f"  vx_max: {last.get('vx_max',0)}m/s\n"
        f"  Status: {'HOAN THANH' if last.get('iter',0)>=last.get('total',3051) else 'CHUA XONG'}\n"
        f"\nSIMULATION: {'DANG CHAY' if _is_sim_running() else 'CHUA CHAY'}\n"
        f"  Run bat: {RUN_BAT}\n"
        f"\nMCP TOOLS (32 tools):\n"
        f"  A. Simulation: start/stop/status/pause/speed\n"
        f"  B. Robot: velocity/stop/reset/mode/push/recovery_test\n"
        f"  C. Telemetry: state/biomechanics/imu/gait/power\n"
        f"  D. Checkpoint: list/info/weights/compare/timeline/export\n"
        f"  E. Training: log/metrics/progress/curriculum/curve/search\n"
        f"  F. Advanced: obs_debug/sim2real/screenshot/project_info"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
async def _amain():
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    async with stdio_server() as (rd, wr):
        await mcp.run(rd, wr, mcp.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(_amain())
