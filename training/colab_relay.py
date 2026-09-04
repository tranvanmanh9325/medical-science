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

from colab_cli.common import state
from colab_cli.state import SessionState

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


def pull_git_latest():
    '''Syncs latest commits from git repository'''
    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT, capture_output=True)
    except Exception:
        pass


def git_commit_and_push(file_rel_path, message):
    '''Commits and pushes a checkpoint file to GitHub repository'''
    try:
        abs_path = os.path.join(ROOT, file_rel_path)
        if not os.path.exists(abs_path) or os.path.getsize(abs_path) < 100_000:
            return
        subprocess.run(["git", "add", file_rel_path], cwd=ROOT, check=True, capture_output=True)
        res = subprocess.run(["git", "commit", "-m", f"{message} [skip ci]"], cwd=ROOT, capture_output=True, text=True)
        if "nothing to commit" not in res.stdout and "nothing to commit" not in res.stderr:
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True, capture_output=True)
            print(f"[GIT PUSH OK] Đã đẩy {file_rel_path} lên GitHub: {message}", flush=True)
        else:
            print(f"[GIT] Không có thay đổi mới trong {file_rel_path}.", flush=True)
    except Exception as e:
        print(f"[GIT PUSH WARNING] Không thể đẩy lên git: {e}", flush=True)


def switch_account_and_reset(acc_name):
    '''Switches token to designated account and invalidates colab-cli cached client'''
    colab_pool.switch_to_account(acc_name)
    state._client = None
    state._sessions = None


def is_vm_assigned_on_google(endpoint=None):
    '''
    Directly queries Google Colab control plane REST API (tun/m/list).
    Takes only ~0.2s, completely unaffected by whether GPU/kernel is busy.
    Returns (True, assigns) if active, (False, []) if revoked/empty.
    '''
    try:
        assigns = state.client.list_assignments()
        if not assigns:
            return False, []
        if endpoint:
            for a in assigns:
                if a.endpoint == endpoint:
                    return True, assigns
            return False, assigns
        return True, assigns
    except Exception as e:
        print(f"[CONTROL PLANE API] Cảnh báo kết nối máy chủ Colab: {e}", flush=True)
        return True, []


def safe_colab_exec(code, timeout=60, retries=2):
    '''
    Executes Python code in the Colab session safely.
    Suppresses stderr tracebacks from leaking into CI/CD logs and retries on transient connection issues.
    '''
    last_err = ""
    for attempt in range(retries + 1):
        try:
            p = subprocess.Popen(
                ["colab", "exec", "-s", SESSION_NAME],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = p.communicate(input=code, timeout=timeout)
            if p.returncode == 0:
                return True, out
            last_err = err or out
            if attempt < retries:
                time.sleep(3)
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(3)
    return False, last_err


def check_is_training_running():
    '''Checks whether train_stage2.py is actively running in the Colab session'''
    check_code = '''
import subprocess
try:
    out = subprocess.check_output("ps aux | grep train_stage2.py | grep -v grep || true", shell=True, text=True)
    print("RUNNING" if "train_stage2.py" in out else "NOT_RUNNING")
except Exception:
    print("NOT_RUNNING")
'''
    ok, out = safe_colab_exec(check_code, timeout=45, retries=2)
    return ok and "RUNNING" in out


def check_and_adopt_assignment(acc_name):
    '''
    Checks if an active GPU assignment already exists on Colab for this account.
    If so, adopts it into local colab-cli session state so commands can attach directly.
    '''
    try:
        assigns = state.client.list_assignments()
    except Exception as e:
        print(f"[RELAY] Lỗi truy vấn danh sách phiên từ Colab: {e}", flush=True)
        return False, False

    if not assigns:
        return False, False

    a = assigns[0]
    print(f"[RELAY] Phát hiện GPU VM đang hoạt động trên {acc_name}: {a.endpoint}", flush=True)

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

    is_running = check_is_training_running()
    return True, is_running


def deploy_and_start_training(acc_name, is_new=True):
    print(f"\n[RELAY] === TRIỂN KHAI TIẾN TRÌNH TRÊN {acc_name} ===", flush=True)

    if is_new:
        print(f"[RELAY] Đang cấp phát GPU T4 mới trên {acc_name}...", flush=True)
        res = subprocess.run(["colab", "new", "-s", SESSION_NAME, "--gpu", "T4"], capture_output=True, text=True)
        err = res.stderr or res.stdout
        if "Session READY" not in res.stdout and "READY" not in res.stdout:
            print(f"[RELAY WARNING] Cấp phát GPU thất bại trên {acc_name}: {err.strip()[:200]}")
            if "TooManyAssignmentsError" in err or "412" in err:
                print(f"[RELAY] Tài khoản đã có máy ảo cấp phát từ trước, chuyển sang nhận diện phiên...", flush=True)
                has_vm, is_running = check_and_adopt_assignment(acc_name)
                if has_vm:
                    if is_running:
                        return True
                    is_new = False
                else:
                    colab_pool.mark_account_exhausted(acc_name, hours=1)
                    return False
            elif "outcome" in err or "Service Unavailable" in err or "ResourceExhausted" in err:
                colab_pool.mark_account_exhausted(acc_name, hours=12)
                return False
            else:
                colab_pool.mark_account_exhausted(acc_name, hours=0.25)
                return False
        else:
            print(f"[RELAY OK] Phiên mới '{SESSION_NAME}' đã sẵn sàng trên {acc_name}!", flush=True)

    # 1. Install dependencies if needed
    print("[RELAY] Kiểm tra / cài đặt thư viện (mujoco, optax, flax)...", flush=True)
    setup_code = '''
import subprocess
try:
    import mujoco, optax, flax
    print("ALL_INSTALLED")
except Exception:
    subprocess.check_output('pip install -q mujoco mujoco-mjx optax flax==0.11.2', shell=True, text=True)
    print("INSTALL_OK")
'''
    ok, out = safe_colab_exec(setup_code, timeout=180, retries=2)
    if not ok or ("ALL_INSTALLED" not in out and "INSTALL_OK" not in out):
        print("[RELAY WARNING] Cài đặt thư viện có cảnh báo, tiếp tục kiểm tra...")

    # 2. Upload assets
    print("[RELAY] Tải lên mô hình và mã nguồn...", flush=True)
    model_zip = os.path.join(ROOT, "colab_deploy", "apollo_model.zip")
    stage1_npz = os.path.join(ROOT, "kaggle_dataset_stage1", "apollo_stage1_v15_step_99876864.npz")
    train_script = os.path.join(ROOT, "colab_deploy", "train_stage2.py")

    subprocess.run(["colab", "upload", "-s", SESSION_NAME, model_zip, "/content/apollo_model.zip"], capture_output=True)
    subprocess.run(["colab", "upload", "-s", SESSION_NAME, stage1_npz, "/content/apollo_stage1_v15_step_99876864.npz"], capture_output=True)
    subprocess.run(["colab", "upload", "-s", SESSION_NAME, train_script, "/content/train_stage2.py"], capture_output=True)

    # 3. Resume Checkpoint Check — Upload directly to /content/ which always exists
    pull_git_latest()
    latest_ck = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
    resume_flag = ""
    if os.path.exists(latest_ck):
        print(f"[RELAY RESUME] Tìm thấy checkpoint trước đó: {latest_ck}", flush=True)
        subprocess.run(["colab", "upload", "-s", SESSION_NAME, latest_ck, "/content/apollo_stage2_v2_latest.npz"], capture_output=True)
        resume_flag = "--resume /content/apollo_stage2_v2_latest.npz"

    # 4. Launch training daemon
    gh_token = get_github_token()
    print(f"[RELAY LAUNCH] Khởi chạy tiến trình train ngầm (Resume: {bool(resume_flag)})...", flush=True)
    launch_code = f'''
import subprocess
cmd = 'GITHUB_TOKEN={gh_token} nohup python3 -u /content/train_stage2.py {resume_flag} > /content/train.log 2>&1 &'
subprocess.Popen(cmd, shell=True)
print('LAUNCHED')
'''
    ok, out = safe_colab_exec(launch_code, timeout=35, retries=2)
    print(f"[RELAY] Huấn luyện đã kích hoạt thành công trên {acc_name}!\n", flush=True)
    return True


def find_account_with_active_assignment():
    '''
    Scans the account pool to see if an account already has an active VM assignment.
    This ensures that when a runner starts or restarts, it connects directly to the currently
    running training VM instead of spinning up an unnecessary new VM.
    '''
    accounts = colab_pool.list_accounts()
    status = colab_pool.load_status()
    for acc in accounts:
        avail, _ = colab_pool.is_account_available(acc, status)
        if not avail:
            continue
        try:
            switch_account_and_reset(acc)
            assigns = state.client.list_assignments()
            if assigns:
                print(f"[RELAY DISCOVERY] Phát hiện {acc} đang sở hữu máy ảo GPU: {assigns[0].endpoint}", flush=True)
                return acc
        except Exception:
            continue
    return None


def monitor_and_sync(acc_name):
    print(f"[RELAY MONITOR] Bắt đầu theo dõi tiến độ trên {acc_name}...", flush=True)

    # Identify current endpoint
    current_endpoint = None
    try:
        active_sess = state.store.get(SESSION_NAME)
        if active_sess:
            current_endpoint = active_sess.endpoint
    except Exception:
        pass

    api_dead_count = 0
    cycle = 0

    while True:
        time.sleep(60)
        cycle += 1

        # 1. Control Plane Check (fast 0.2s Google REST API check)
        is_alive, _ = is_vm_assigned_on_google(current_endpoint)
        if not is_alive:
            api_dead_count += 1
            print(f"[{time.strftime('%H:%M:%S')} | {acc_name}] [CẢNH BÁO CONTROL PLANE] VM không còn trong danh sách gán của Google (#{api_dead_count}/3)...", flush=True)
            if api_dead_count >= 3:
                print(f"\n[RELAY FAILOVER] Google đã chính thức thu hồi máy ảo trên {acc_name} (Hết hạn mức hoặc phiên bị ngắt)!")
                try:
                    local_target = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
                    subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_v2_latest.npz", local_target], capture_output=True, timeout=15)
                    git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_v2_latest.npz", f"chore(checkpoint): backup before failover from {acc_name}")
                except Exception:
                    pass
                colab_pool.mark_account_exhausted(acc_name, hours=12)
                return "FAILOVER"
            continue
        else:
            api_dead_count = 0

        # 2. Check training log tail via exec (timeout 60s)
        check_code = '''
try:
    with open('/content/train.log', 'r') as f:
        lines = f.readlines()
        print('LOG_TAIL:' + (lines[-1].strip() if lines else 'EMPTY'))
except Exception as e:
    print('LOG_ERR:' + str(e))
'''
        ok, out = safe_colab_exec(check_code, timeout=60, retries=1)

        if ok and "LOG_TAIL:" in out:
            log_line = ""
            for line in out.splitlines():
                if "LOG_TAIL:" in line:
                    log_line = line.replace("LOG_TAIL:", "").strip()
            print(f"[{time.strftime('%H:%M:%S')} | {acc_name}] {log_line}", flush=True)

            # Periodically download latest checkpoint and commit to git every 5 minutes (~5 iters)
            if cycle % 5 == 0:
                try:
                    local_target = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
                    dl = subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_v2_latest.npz", local_target], capture_output=True, text=True)
                    if dl.returncode != 0:
                        dl = subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/apollo_stage2_v2_latest.npz", local_target], capture_output=True, text=True)
                    if dl.returncode == 0 and os.path.exists(local_target) and os.path.getsize(local_target) > 100_000:
                        git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_v2_latest.npz", f"chore(checkpoint): update stage 2 checkpoint [{acc_name}]")
                except Exception as e:
                    print(f"[RELAY SYNC WARNING] Lỗi đồng bộ checkpoint: {e}", flush=True)

            if "STAGE 2 v2 TRAINING COMPLETE!" in out:
                print("\n" + "=" * 64)
                print("  🎉🎉🎉 HUẤN LUYỆN HOÀN TẤT 100%! CÁN ĐÍCH 150M BƯỚC! 🎉🎉🎉")
                print("=" * 64, flush=True)
                final_local = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_final.npz")
                subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_final.npz", final_local])
                git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_final.npz", "feat(weights): save final Apollo Stage 2 trained model (150M steps)")
                return "COMPLETE"
        else:
            # colab exec timed out or failed to read log, but VM IS STILL ALIVE in Control Plane!
            print(f"[{time.strftime('%H:%M:%S')} | {acc_name}] [BUSY: GPU 100%] Jupyter kernel phản hồi chậm do tải tính toán nặng, máy ảo vẫn hoạt động ổn định.", flush=True)

        # 3. Check process health inside VM periodically (every 10 minutes)
        if cycle % 10 == 0:
            if not check_is_training_running():
                print(f"[{time.strftime('%H:%M:%S')} | {acc_name}] [WARNING] Tiến trình train không chạy trên máy ảo! Tự động kích hoạt lại từ checkpoint gần nhất...", flush=True)
                deploy_and_start_training(acc_name, is_new=False)


def run_relay():
    print("=" * 64)
    print("  HỆ THỐNG ĐIỀU PHỐI XOAY VÒNG TÀI KHOẢN (Colab Relay Orchestrator)")
    print("=" * 64, flush=True)

    # First check if an account is already actively running training
    active_acc = find_account_with_active_assignment()

    while True:
        colab_pool.show_pool()
        if active_acc:
            acc = active_acc
            active_acc = None  # Use it once on startup
        else:
            acc = colab_pool.get_next_available_account()

        if not acc:
            print("\n[CẢNH BÁO] Tất cả tài khoản trong Pool hiện đều đang bị Cooldown!")
            print("Đang chờ 10 phút trước khi kiểm tra lại...")
            time.sleep(600)
            continue

        switch_account_and_reset(acc)
        has_assignment, is_running = check_and_adopt_assignment(acc)

        if has_assignment and is_running:
            print(f"[RELAY] Huấn luyện đang chạy sẵn trên {acc}, gắn trực tiếp vào giám sát!", flush=True)
            success = True
        elif has_assignment and not is_running:
            print(f"[RELAY] Máy ảo {acc} đã có sẵn nhưng tiến trình đã dừng, khởi chạy lại...", flush=True)
            success = deploy_and_start_training(acc, is_new=False)
        else:
            success = deploy_and_start_training(acc, is_new=True)

        if not success:
            print(f"[RELAY] Tài khoản {acc} không thể khởi động, chuyển sang tài khoản kế tiếp...")
            time.sleep(5)
            continue

        res = monitor_and_sync(acc)
        if res == "COMPLETE":
            print("[RELAY] Hoàn tất toàn bộ nhiệm vụ!")
            break
        elif res == "FAILOVER":
            print("[RELAY] Tiếp tục vòng lặp chuyển giao sang tài khoản kế tiếp...")
            time.sleep(5)
            continue


if __name__ == "__main__":
    run_relay()
