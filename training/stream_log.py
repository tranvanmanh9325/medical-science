"""
Apollo Stage 2 Training — Live Log Streamer (Final Model Only)
Usage:
    python training/stream_log.py
"""

import os
import sys
import time
import json
import re
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CKPT_DIR = os.path.join(ROOT, "colab_output", "checkpoints_stage2")
SESSION_NAME = "stage2-train"

from colab_cli.common import state
from colab_cli.contents import ContentsClient


def run_colab(code_str, timeout=15):
    try:
        cmd = ["colab", "exec", "-s", SESSION_NAME]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = p.communicate(input=code_str, timeout=timeout)
        return out
    except Exception as e:
        return ""


def download_final_model():
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    print("\n[COLAB] Đang tải duy nhất file hoàn thiện cuối cùng về máy...")
    code = """
import glob, os, shutil
ckpts = sorted(glob.glob('/content/checkpoints/*.npz'))
if ckpts:
    latest = ckpts[-1]
    final_path = '/content/checkpoints/apollo_stage2_final.npz'
    shutil.copy2(latest, final_path)
    print(final_path)
"""
    raw = run_colab(code).strip()
    match = re.search(r"(/content/checkpoints/apollo_stage2_final\.npz)", raw)
    if match:
        remote_file = match.group(1)
        local_target = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_final.npz")
        subprocess.run(["colab", "download", "-s", SESSION_NAME, remote_file, local_target])
        print(f"[COLAB] ĐÃ TẢI XONG FILE HOÀN CHỈNH DUY NHẤT: {local_target}\n", flush=True)


def ensure_session_attached():
    '''
    Checks if SESSION_NAME is attached locally.
    If not, iterates through colab_accounts, finds the account holding an active GPU VM,
    switches to it, and adopts the session so streaming works seamlessly.
    '''
    s = state.store.get(SESSION_NAME)
    if s:
        return True

    print("[STREAM] Đang tự động tìm kiếm phiên huấn luyện trên các tài khoản Google Colab...", flush=True)
    sys.path.insert(0, os.path.join(ROOT, "training"))
    import colab_pool
    from colab_cli.state import SessionState

    for acc in colab_pool.list_accounts():
        try:
            colab_pool.switch_to_account(acc)
            state._client = None
            state._sessions = None
            assigns = state.client.list_assignments()
            if assigns:
                a = assigns[0]
                print(f"[STREAM] Tìm thấy máy ảo đang hoạt động trên {acc}: {a.endpoint}", flush=True)
                s = SessionState(
                    name=SESSION_NAME,
                    token=a.runtime_proxy_info.token,
                    url=a.runtime_proxy_info.url,
                    endpoint=a.endpoint,
                    variant="GPU",
                    accelerator="T4",
                )
                state.store.add(s)
                state._sessions = None
                return True
        except Exception:
            continue
    return False


def stream():
    print("=" * 64)
    print("  APOLLO STAGE 2 — LIVE STREAMING LOG (Final Checkpoint Only)")
    print(f"  Session: {SESSION_NAME} | Tesla T4 GPU (Google Colab)")
    print("  Nhấn Ctrl+C để thoát bất cứ lúc nào (server vẫn chạy tiếp)")
    print("=" * 64, flush=True)

    # Check and automatically adopt session across accounts
    if not ensure_session_attached():
        print(f"\n[LỖI] Phiên '{SESSION_NAME}' hiện KHÔNG còn hoạt động trên các tài khoản Colab.")
        print("Không thể stream log. Hãy kiểm tra trạng thái GitHub Actions Cloud Runner.")
        return

    s = state.store.get(SESSION_NAME)
    if not s:
        print("[LỖI] Không tìm thấy phiên để kết nối.")
        return

    contents = ContentsClient(s)
    current_offset = 0

    # Fetch initial log content instantly via HTTP REST API
    try:
        data = contents._request("GET", "content/train.log", params={"content": "1"})
        full_text = data.get("content", "")
        if full_text:
            print(full_text, end="", flush=True)
            current_offset = len(full_text)
    except Exception as e:
        print(f"[STREAM INFO] Đang chờ file log khởi tạo trên máy ảo...")

    while True:
        try:
            data = contents._request("GET", "content/train.log", params={"content": "1"})
            full_text = data.get("content", "")
            if len(full_text) > current_offset:
                new_text = full_text[current_offset:]
                current_offset = len(full_text)
                print(new_text, end="", flush=True)
                if "STAGE 2 v2 TRAINING COMPLETE!" in new_text:
                    print("\n" + "=" * 64)
                    print("  🎉🎉🎉 HUẤN LUYỆN STAGE 2 ĐÃ HOÀN TẤT 100%! 🎉🎉🎉")
                    print("  👉 Hãy báo lại cho AI: 'Train xong rồi, hãy check giúp tôi'")
                    print("=" * 64 + "\n", flush=True)
                    download_final_model()
                    return

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n[INFO] Đã tạm dừng xem log trên terminal. Máy chủ Colab vẫn đang tiếp tục train!")
            break
        except Exception:
            time.sleep(5)


if __name__ == "__main__":
    stream()
