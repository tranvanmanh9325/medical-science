'''
Apollo Colab Multi-Account Pool Manager
Manages multiple Google Colab account tokens and handles switching between accounts.
'''

import os
import sys
import json
import time
import shutil
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNTS_DIR = os.path.join(ROOT, "training", "colab_accounts")
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "colab-cli")
TOKEN_TARGET = os.path.join(CONFIG_DIR, "token.json")
STATUS_FILE = os.path.join(ACCOUNTS_DIR, "pool_status.json")

os.makedirs(ACCOUNTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def load_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_status(status):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def list_accounts():
    accounts = []
    if os.path.exists(ACCOUNTS_DIR):
        for f in os.listdir(ACCOUNTS_DIR):
            if f.endswith(".json") and f != "pool_status.json":
                acc_name = f.replace(".json", "")
                accounts.append(acc_name)
    return sorted(accounts)


def is_account_available(acc_name, status):
    info = status.get(acc_name, {})
    cooldown_until = info.get("cooldown_until", 0)
    now = time.time()
    if now < cooldown_until:
        rem_min = (cooldown_until - now) / 60
        return False, f"Cooldown còn {rem_min:.0f} phút"
    return True, "Sẵn sàng"


def mark_account_exhausted(acc_name, hours=12):
    status = load_status()
    cooldown_until = time.time() + (hours * 3600)
    status[acc_name] = {
        "cooldown_until": cooldown_until,
        "exhausted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "GPU Quota Limit (503 / outcome: 2)"
    }
    save_status(status)
    print(f"[POOL] Đã đánh dấu {acc_name} tạm dừng trong {hours} tiếng.")


def mark_account_active(acc_name):
    status = load_status()
    if acc_name in status:
        del status[acc_name]
        save_status(status)


def switch_to_account(acc_name):
    token_src = os.path.join(ACCOUNTS_DIR, f"{acc_name}.json")
    if not os.path.exists(token_src):
        raise FileNotFoundError(f"Không tìm thấy token cho {acc_name}")
    shutil.copy2(token_src, TOKEN_TARGET)
    print(f"[POOL] Đã chuyển đổi token sang: {acc_name}")
    return True


def get_next_available_account(exclude=None):
    accounts = list_accounts()
    status = load_status()
    for acc in accounts:
        if exclude and acc == exclude:
            continue
        avail, reason = is_account_available(acc, status)
        if avail:
            return acc
    return None


def show_pool():
    accounts = list_accounts()
    status = load_status()
    print("=" * 64)
    print("  HỒ BƠI TÀI KHOẢN GOOGLE COLAB (Account Pool)")
    print("=" * 64)
    if not accounts:
        print("  Chưa có tài khoản nào được lưu!")
        return
    for acc in accounts:
        avail, msg = is_account_available(acc, status)
        tag = "[READY]" if avail else "[COOLDOWN]"
        print(f"  {tag:12s} {acc:15s} -> {msg}")
    print("=" * 64)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            show_pool()
        elif cmd == "switch" and len(sys.argv) > 2:
            switch_to_account(sys.argv[2])
        elif cmd == "exhaust" and len(sys.argv) > 2:
            mark_account_exhausted(sys.argv[2])
    else:
        show_pool()
