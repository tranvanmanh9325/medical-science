'''
Apollo Colab Multi-Account Failover & Relay Orchestrator
Continuously monitors training on Google Colab. If an account hits GPU limits (503 / 404),
it automatically switches credentials to the next available account in the pool,
pulls the latest checkpoint from GitHub, and resumes training seamlessly.
'''

import os
import sys
import time
import json
import shutil
import subprocess
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import colab_pool

SESSION_NAME = "stage2-train"
LOCAL_CKPT_DIR = os.path.join(ROOT, "colab_output", "checkpoints_stage2")
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

def get_github_token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                token = res.stdout.strip()
        except Exception:
            pass
    return token


def get_latest_checkpoint_from_git():
    '''Syncs latest checkpoint committed to git repository'''
    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT, capture_output=True)
    except Exception:
        pass
    latest_ck = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
    if os.path.exists(latest_ck):
        return latest_ck
    return None


def deploy_and_start_training(acc_name):
    print(f"\n[RELAY] === TRIỂN KHAI TIẾN TRÌNH TRÊN {acc_name} ===", flush=True)
    colab_pool.switch_to_account(acc_name)

    # 1. Spawn session
    print(f"[RELAY] Đang cấp phát GPU T4 mới trên {acc_name}...", flush=True)
    res = subprocess.run(["colab", "new", "-s", SESSION_NAME, "--gpu", "T4"], capture_output=True, text=True)
    if "Session READY" not in res.stdout and "READY" not in res.stdout:
        print(f"[RELAY WARNING] Cấp phát GPU thất bại trên {acc_name}: {res.stderr or res.stdout}")
        if "outcome" in res.stdout or "outcome" in res.stderr or "Service Unavailable" in res.stderr:
            colab_pool.mark_account_exhausted(acc_name, hours=12)
        return False

    print(f"[RELAY OK] Phiên '{SESSION_NAME}' đã sẵn sàng trên {acc_name}!", flush=True)

    # 2. Install dependencies
    print("[RELAY] Cài đặt thư viện (mujoco, optax, flax)...", flush=True)
    setup_code = '''
import subprocess
subprocess.check_output('pip install -q mujoco mujoco-mjx optax flax==0.11.2', shell=True, text=True)
print('INSTALL_OK')
'''
    p = subprocess.Popen(["colab", "exec", "-s", SESSION_NAME], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input=setup_code)
    if "INSTALL_OK" not in out:
        print("[RELAY WARNING] Cài đặt thư viện có cảnh báo, tiếp tục kiểm tra...")

    # 3. Upload assets
    print("[RELAY] Tải lên mô hình và mã nguồn...", flush=True)
    model_zip = os.path.join(ROOT, "colab_deploy", "apollo_model.zip")
    stage1_npz = os.path.join(ROOT, "kaggle_dataset_stage1", "apollo_stage1_v15_step_99876864.npz")
    train_script = os.path.join(ROOT, "colab_deploy", "train_stage2.py")

    subprocess.run(["colab", "upload", "-s", SESSION_NAME, model_zip, "/content/apollo_model.zip"], capture_output=True)
    subprocess.run(["colab", "upload", "-s", SESSION_NAME, stage1_npz, "/content/apollo_stage1_v15_step_99876864.npz"], capture_output=True)
    subprocess.run(["colab", "upload", "-s", SESSION_NAME, train_script, "/content/train_stage2.py"], capture_output=True)

    # 4. Check for resume checkpoint
    latest_ck = get_latest_checkpoint_from_git()
    resume_flag = ""
    if latest_ck and os.path.exists(latest_ck):
        print(f"[RELAY RESUME] Tìm thấy checkpoint trước đó: {latest_ck}", flush=True)
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, latest_ck, "/content/checkpoints/apollo_stage2_v2_latest.npz"], capture_output=True)
        resume_flag = "--resume /content/checkpoints/apollo_stage2_v2_latest.npz"

    # 5. Launch training daemon
    gh_token = get_github_token()
    print(f"[RELAY LAUNCH] Khởi chạy tiến trình train ngầm (Resume: {bool(resume_flag)})...", flush=True)
    launch_code = f'''
import subprocess
cmd = 'GITHUB_TOKEN={gh_token} nohup python3 -u /content/train_stage2.py {resume_flag} > /content/train.log 2>&1 &'
subprocess.Popen(cmd, shell=True)
print('LAUNCHED')
'''
    p = subprocess.Popen(["colab", "exec", "-s", SESSION_NAME], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _ = p.communicate(input=launch_code)
    print(f"[RELAY] Huấn luyện đã kích hoạt thành công trên {acc_name}!\n", flush=True)
    return True


def run_relay():
    print("=" * 64)
    print("  HỆ THỐNG ĐIỀU PHỐI XOAY VÒNG TÀI KHOẢN (Colab Relay)")
    print("=" * 64, flush=True)

    while True:
        colab_pool.show_pool()
        acc = colab_pool.get_next_available_account()
        if not acc:
            print("\n[CẢNH BÁO] Tất cả tài khoản trong Pool hiện đều đang bị Cooldown!")
            print("Đang chờ 10 phút trước khi kiểm tra lại...")
            time.sleep(600)
            continue

        # Check if session is already running on this account
        colab_pool.switch_to_account(acc)
        check_sess = subprocess.run(["colab", "sessions"], capture_output=True, text=True)
        if SESSION_NAME in check_sess.stdout:
            print(f"[RELAY] Phiên '{SESSION_NAME}' đang chạy sẵn trên {acc}, gắn trực tiếp vào giám sát!", flush=True)
            success = True
        else:
            success = deploy_and_start_training(acc)
        if not success:
            print(f"[RELAY] Tài khoản {acc} không thể khởi động, chuyển sang tài khoản kế tiếp...")
            time.sleep(5)
            continue

        # Monitoring loop for this active account
        print(f"[RELAY MONITOR] Bắt đầu theo dõi tiến độ trên {acc}...", flush=True)
        fail_count = 0

        while True:
            time.sleep(60)
            check_code = '''
try:
    with open('/content/train.log', 'r') as f:
        lines = f.readlines()
        print('LOG_TAIL:' + (lines[-1].strip() if lines else 'EMPTY'))
except Exception as e:
    print('LOG_ERR:' + str(e))
'''
            try:
                p = subprocess.Popen(["colab", "exec", "-s", SESSION_NAME], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, err = p.communicate(input=check_code, timeout=25)
            except Exception as e:
                out = ""

            if "LOG_TAIL:" in out:
                fail_count = 0
                log_line = ""
                for line in out.splitlines():
                    if "LOG_TAIL:" in line:
                        log_line = line.replace("LOG_TAIL:", "").strip()
                print(f"[{time.strftime('%H:%M:%S')} | {acc}] {log_line}", flush=True)

                if "STAGE 2 v2 TRAINING COMPLETE!" in out:
                    print("\n" + "=" * 64)
                    print("  🎉🎉🎉 HUẤN LUYỆN HOÀN TẤT 100%! 🎉🎉🎉")
                    print("=" * 64, flush=True)
                    subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_final.npz", os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_final.npz")])
                    return
            else:
                fail_count += 1
                print(f"[{time.strftime('%H:%M:%S')} | {acc}] Cảnh báo: Mất kết nối #{fail_count}/3...", flush=True)
                if fail_count >= 3:
                    print(f"\n[RELAY FAILOVER] Phiên trên {acc} đã bị ngắt hoặc chạm hạn mức!")
                    colab_pool.mark_account_exhausted(acc, hours=12)
                    # Pull any new checkpoint from GitHub before failover
                    get_latest_checkpoint_from_git()
                    break


if __name__ == "__main__":
    run_relay()
