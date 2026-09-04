"""
Monitor Apollo Stage 2 Training on Colab VM & auto-download checkpoints.
Usage:
    python training/monitor_colab.py          # check status + show log + download new ckpts
    python training/monitor_colab.py --watch  # continuous monitoring loop
"""

import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CKPT_DIR = os.path.join(ROOT, "colab_output", "checkpoints_stage2")
SESSION_NAME = "stage2-train"


def run_colab_code(code_str):
    cmd = ["colab", "exec", "-s", SESSION_NAME]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(input=code_str)
    return out


def check_status():
    print("=" * 64)
    print(f"  APOLLO STAGE 2 TRAINING MONITOR (Session: {SESSION_NAME})")
    print("=" * 64)
    
    # 1. Check GPU & PID
    py_code = """
import subprocess
print('--- GPU STATUS ---')
print(subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader', shell=True, text=True).strip())
print('--- PROCESS STATUS ---')
print(subprocess.check_output('ps aux | grep train_stage2 | grep -v grep', shell=True, text=True).strip())
print('--- RECENT LOGS ---')
with open('/content/train.log', 'r') as f:
    lines = f.readlines()
    print(''.join(lines[-10:]).strip())
print('--- CHECKPOINTS ON VM ---')
import glob, os
ckpts = sorted(glob.glob('/content/checkpoints/*.npz'))
for c in ckpts:
    print(f'{c} ({os.path.getsize(c)} bytes)')
"""
    output = run_colab_code(py_code)
    print(output)
    
    # 2. Check and download checkpoints (Only when explicitly requested)
    if "--download" in sys.argv:
        os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
        list_code = "import glob; print(','.join(glob.glob('/content/checkpoints/*.npz')))"
        ckpts_raw = run_colab_code(list_code).strip()
        if ckpts_raw:
            for remote_ck in ckpts_raw.split(","):
                remote_ck = remote_ck.strip()
                if remote_ck.endswith(".npz"):
                    fname = os.path.basename(remote_ck)
                    local_ck = os.path.join(LOCAL_CKPT_DIR, fname)
                    if not os.path.exists(local_ck):
                        print(f"[DOWNLOAD] Found new checkpoint: {fname}")
                        subprocess.run(["colab", "download", "-s", SESSION_NAME, remote_ck, local_ck])
                        print(f"  -> Saved to {local_ck}")
                    else:
                        print(f"[OK] Checkpoint already downloaded: {fname}")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        print("[MONITOR] Starting continuous monitor (Ctrl+C to exit)...")
        while True:
            try:
                check_status()
                time.sleep(60)
            except KeyboardInterrupt:
                break
    else:
        check_status()
