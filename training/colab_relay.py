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
from datetime import datetime, timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))


def get_vn_time_str(fmt="%H:%M:%S"):
    '''Returns current timestamp in Vietnam timezone (UTC+7)'''
    return datetime.now(VN_TZ).strftime(fmt)


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import colab_pool

# Compatibility patch for google-colab-cli with jupyter_kernel_client >= 1.0.0
try:
    import jupyter_kernel_client
    if not hasattr(jupyter_kernel_client, "KernelClient") and hasattr(jupyter_kernel_client, "JupyterKernelClient"):
        jupyter_kernel_client.KernelClient = jupyter_kernel_client.JupyterKernelClient
except Exception:
    pass

from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.contents import ContentsClient

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


def is_gpu_assignment(assignment):
    '''
    Strictly verifies whether a Colab assignment is an active GPU VM (T4, L4, A100, H100, G4).
    Strictly rejects CPU (NONE / DEFAULT) assignments.
    '''
    if not assignment:
        return False
    accel = str(getattr(assignment, "accelerator", "")).upper()
    variant = str(getattr(assignment, "variant", "")).upper()

    if "NONE" in accel:
        return False

    return any(g in accel for g in ["T4", "L4", "A100", "H100", "G4"]) or "GPU" in variant or variant == "1"


def purge_non_gpu_assignments(acc_name, assigns):
    '''
    Releases any CPU VM assignments on the account so they do not block allocating a GPU VM.
    '''
    for a in assigns:
        if not is_gpu_assignment(a):
            try:
                print(f"[RELAY PURGE] Máy ảo {a.endpoint} trên {acc_name} là CPU ({a.accelerator}). Đang thu hồi (unassign)...", flush=True)
                state.client.unassign(a.endpoint)
                print(f"[RELAY PURGE OK] Đã giải phóng thành công {a.endpoint} trên {acc_name}.", flush=True)
            except Exception as e:
                print(f"[RELAY PURGE WARNING] Lỗi khi giải phóng {a.endpoint}: {e}", flush=True)


def is_vm_assigned_on_google(endpoint=None, acc_name=None):
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
        err_str = str(e)
        if ("401" in err_str or "auth" in err_str.lower()) and acc_name:
            print(f"[CONTROL PLANE API] Token expired during list_assignments, refreshing for {acc_name}...", flush=True)
            colab_pool.refresh_account_token(acc_name)
            state._client = None
            state._auth_provider = None
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
            except Exception:
                pass
        print(f"[CONTROL PLANE API] Cảnh báo kết nối máy chủ Colab: {e}", flush=True)
        return True, []


def check_and_adopt_assignment(acc_name):
    '''
    Checks if an active GPU assignment already exists on Colab for this account.
    If a non-GPU assignment is found, automatically unassigns it to free the account.
    If a valid GPU exists, adopts it into local colab-cli session state.
    '''
    try:
        assigns = state.client.list_assignments()
    except Exception as e:
        if "401" in str(e) or "auth" in str(e).lower():
            colab_pool.refresh_account_token(acc_name)
            state._client = None
            state._auth_provider = None
            try:
                assigns = state.client.list_assignments()
            except Exception as e2:
                print(f"[RELAY] Lỗi truy vấn danh sách phiên sau khi refresh: {e2}", flush=True)
                return False, False
        else:
            print(f"[RELAY] Lỗi truy vấn danh sách phiên từ Colab: {e}", flush=True)
            return False, False

    if not assigns:
        return False, False

    # Purge any non-GPU (CPU) assignments
    purge_non_gpu_assignments(acc_name, assigns)

    # Find valid GPU assignment
    gpu_assign = None
    for a in assigns:
        if is_gpu_assignment(a):
            gpu_assign = a
            break

    if not gpu_assign:
        return False, False

    print(f"[RELAY] Phát hiện GPU VM ({gpu_assign.accelerator}) đang hoạt động trên {acc_name}: {gpu_assign.endpoint}", flush=True)

    accel_name = str(gpu_assign.accelerator).replace("Accelerator.", "")
    if accel_name == "NONE":
        accel_name = "T4"

    s = SessionState(
        name=SESSION_NAME,
        token=gpu_assign.runtime_proxy_info.token,
        url=gpu_assign.runtime_proxy_info.url,
        endpoint=gpu_assign.endpoint,
        variant="GPU",
        accelerator=accel_name,
    )
    state.store.add(s)
    state._sessions = None
    colab_pool.save_account_sessions(acc_name)

    try:
        from colab_cli.commands.session import spawn_keep_alive
        s.keep_alive_pid = spawn_keep_alive(
            gpu_assign.endpoint,
            SESSION_NAME,
            auth_provider=state.auth_provider,
            config_path=state.config_path,
        )
        state.store.add(s)
        colab_pool.save_account_sessions(acc_name)
    except Exception:
        pass

    is_running = check_is_training_running(acc_name)
    return True, is_running


def ensure_session_valid(acc_name):
    '''
    Guarantees that SESSION_NAME exists in state.store and sessions.json.
    If missing (e.g. pruned by colab-cli on transient 401), restores it from
    either backup sessions_{acc_name}.json or re-adopts from Google control plane.
    '''
    s = state.store.get(SESSION_NAME)
    if s:
        return True

    # 1. Try restore from backup file
    if colab_pool.restore_account_sessions(acc_name):
        state._sessions = None
        s = state.store.get(SESSION_NAME)
        if s:
            print(f"[RELAY HEAL] Khôi phục phiên '{SESSION_NAME}' từ file dự phòng cho {acc_name}!", flush=True)
            return True

    # 2. Try re-adopt from Google assignments
    adopted, _ = check_and_adopt_assignment(acc_name)
    if adopted:
        print(f"[RELAY HEAL] Tái kết nối phiên '{SESSION_NAME}' từ Google control plane cho {acc_name}!", flush=True)
        return True

    return False


def safe_colab_exec(code, timeout=90, retries=2, acc_name=None):
    '''
    Executes Python code in the Colab session safely.
    Explicitly passes --timeout to colab-cli to prevent premature 30s timeouts.
    If session is missing or pruned, auto-restores before running.
    '''
    if acc_name:
        ensure_session_valid(acc_name)

    last_err = ""
    for attempt in range(retries + 1):
        try:
            p = subprocess.Popen(
                ["colab", "exec", "--timeout", str(timeout), "-s", SESSION_NAME],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = p.communicate(input=code, timeout=timeout + 5)
            if p.returncode == 0:
                return True, out
            last_err = err or out
            if "Session" in last_err and "not found" in last_err.lower() and acc_name:
                print(f"[RELAY HEAL] Phiên bị mất trong safe_colab_exec. Khôi phục lại cho {acc_name}...", flush=True)
                ensure_session_valid(acc_name)
            elif ("401" in last_err or "auth" in last_err.lower()) and acc_name:
                print(f"[RELAY HEAL] Lỗi xác thực trong safe_colab_exec. Refresh token cho {acc_name}...", flush=True)
                colab_pool.refresh_account_token(acc_name)
                state._client = None
                state._auth_provider = None
                ensure_session_valid(acc_name)
            if attempt < retries:
                time.sleep(3)
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(3)
    return False, last_err


def check_is_training_running(acc_name=None):
    '''
    Checks whether train_stage2.py is actively running in the Colab session.
    First performs ultra-fast 0.1s HTTP REST check via train.log metadata and content.
    If train.log contains active training markers, training is confirmed running without
    touching the fragile WebSocket / Jupyter kernel.
    Falls back to ps aux via safe_colab_exec only if train.log is missing.
    '''
    if acc_name:
        ensure_session_valid(acc_name)
    try:
        s = state.store.get(SESSION_NAME)
        if s:
            contents = ContentsClient(s)
            data = contents._request("GET", "content/train.log", params={"content": "1"})
            content = data.get("content", "")
            if content and any(m in content for m in ["steps=", "WALKING", "APOLLO HUMANOID", "RESUME SUCCESS", "[TRANSFER LEARNING]"]):
                return True
    except Exception:
        pass

    check_code = '''
import subprocess
try:
    out = subprocess.check_output("ps aux | grep train_stage2.py | grep -v grep || true", shell=True, text=True)
    print("RUNNING" if "train_stage2.py" in out else "NOT_RUNNING")
except Exception:
    print("NOT_RUNNING")
'''
    ok, out = safe_colab_exec(check_code, timeout=45, retries=2, acc_name=acc_name)
    return ok and "RUNNING" in out


def fetch_remote_train_log(acc_name=None):
    '''
    Directly fetches /content/train.log via Google Colab HTTP REST API (ContentsClient).
    Bypasses Jupyter kernel and WebSocket completely. Takes only ~0.2s and NEVER hangs.
    If session is missing from local store, attempts automatic self-healing by re-adopting.
    '''
    try:
        s = state.store.get(SESSION_NAME)
        if not s and acc_name:
            ensure_session_valid(acc_name)
            s = state.store.get(SESSION_NAME)
        if not s:
            return False, "SESSION_NOT_FOUND"
        contents = ContentsClient(s)
        data = contents._request("GET", "content/train.log", params={"content": "1"})
        return True, data.get("content", "")
    except FileNotFoundError:
        return False, "FILE_NOT_FOUND"
    except Exception as e:
        err_str = str(e)
        if "404" in err_str:
            return False, "FILE_NOT_FOUND"
        elif "401" in err_str or "auth" in err_str.lower() or "unauthorized" in err_str.lower():
            return False, "AUTH_EXPIRED"
        return False, err_str


def deploy_and_start_training(acc_name, is_new=True):
    print(f"\n[RELAY] === TRIỂN KHAI TIẾN TRÌNH TRÊN {acc_name} ===", flush=True)

    if is_new:
        print(f"[RELAY] Đang cấp phát GPU T4 mới trên {acc_name}...", flush=True)
        res = subprocess.run(["colab", "new", "-s", SESSION_NAME, "--gpu", "T4"], capture_output=True, text=True)
        err = res.stderr or res.stdout
        if "Session READY" not in res.stdout and "READY" not in res.stdout:
            print(f"[RELAY WARNING] Cấp phát GPU thất bại trên {acc_name}: {err.strip()[:200]}")
            if "TooManyAssignmentsError" in err or "412" in err:
                print(f"[RELAY] Tài khoản đã có máy ảo cấp phát từ trước, kiểm tra loại máy ảo...", flush=True)
                has_vm, is_running = check_and_adopt_assignment(acc_name)
                if has_vm:
                    if is_running:
                        return True
                    is_new = False
                else:
                    # Non-GPU VM was purged, retry allocating fresh GPU
                    print(f"[RELAY] Đã giải phóng máy ảo CPU cũ, thử cấp phát lại GPU T4 trên {acc_name}...", flush=True)
                    time.sleep(3)
                    res2 = subprocess.run(["colab", "new", "-s", SESSION_NAME, "--gpu", "T4"], capture_output=True, text=True)
                    if "Session READY" in res2.stdout or "READY" in res2.stdout:
                        print(f"[RELAY OK] Phiên mới '{SESSION_NAME}' đã sẵn sàng trên {acc_name} sau khi giải phóng!", flush=True)
                        colab_pool.save_account_sessions(acc_name)
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
            colab_pool.save_account_sessions(acc_name)
    else:
        ensure_session_valid(acc_name)

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
    ok, out = safe_colab_exec(setup_code, timeout=180, retries=2, acc_name=acc_name)
    if not ok or ("ALL_INSTALLED" not in out and "INSTALL_OK" not in out):
        print(f"[RELAY WARNING] Cài đặt thư viện có cảnh báo ({out.strip()[:100]}), tiếp tục kiểm tra...")

    # 2. Upload assets
    print("[RELAY] Tải lên mô hình và mã nguồn...", flush=True)
    ensure_session_valid(acc_name)
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
subprocess.Popen(cmd, shell=True, start_new_session=True)
print('LAUNCHED_SUCCESSFULLY')
'''
    ok, out = safe_colab_exec(launch_code, timeout=45, retries=3, acc_name=acc_name)
    if not ok or "LAUNCHED_SUCCESSFULLY" not in out:
        print(f"[RELAY ERROR] Lệnh khởi chạy train thất bại trên {acc_name}: {out}", flush=True)
        return False

    time.sleep(5)
    if not check_is_training_running(acc_name):
        print(f"[RELAY ERROR] Tiến trình không còn chạy ngay sau khi kích hoạt trên {acc_name}!", flush=True)
        ok_log, crash_log = fetch_remote_train_log(acc_name)
        if ok_log and crash_log:
            print(f"[RELAY CRASH LOG]\n{crash_log[-1000:]}", flush=True)
        return False

    print(f"[RELAY] Huấn luyện đã kích hoạt thành công trên {acc_name}!\n", flush=True)
    return True


def find_account_with_active_assignment():
    '''
    Scans the account pool to see if an account already has an active GPU VM assignment.
    Auto-purges any CPU assignments encountered so accounts are immediately clean for GPU allocation.
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
            if not assigns:
                continue

            # Check and purge CPU assignments
            purge_non_gpu_assignments(acc, assigns)

            for a in assigns:
                if is_gpu_assignment(a):
                    print(f"[RELAY DISCOVERY] Phát hiện {acc} đang sở hữu máy ảo GPU ({a.accelerator}): {a.endpoint}", flush=True)
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
    last_progress_time = time.time()
    launch_time = time.time()
    consecutive_log_fails = 0
    last_printed_line_idx = 0

    while True:
        time.sleep(60)
        cycle += 1

        # Proactive OAuth token refresh every 15 minutes (900 seconds)
        if cycle % 15 == 0:
            print(f"[{get_vn_time_str()} | {acc_name}] [PROACTIVE AUTH] Tự động gia hạn OAuth token...", flush=True)
            colab_pool.refresh_account_token(acc_name)
            colab_pool.save_account_sessions(acc_name)
            state._client = None
            state._auth_provider = None

        # Keep alive ping to control plane on every cycle
        if current_endpoint:
            try:
                state.client.keep_alive_assignment(current_endpoint)
            except Exception:
                pass

        # 1. Control Plane Check (fast 0.2s Google REST API check)
        is_alive, _ = is_vm_assigned_on_google(current_endpoint, acc_name=acc_name)
        if not is_alive:
            api_dead_count += 1
            print(f"[{get_vn_time_str()} | {acc_name}] [CẢNH BÁO CONTROL PLANE] VM không còn trong danh sách gán của Google (#{api_dead_count}/3)...", flush=True)
            if api_dead_count >= 3:
                print(f"\n[RELAY FAILOVER] Google đã chính thức thu hồi máy ảo trên {acc_name} (Hết hạn mức hoặc phiên bị ngắt)!")
                try:
                    ensure_session_valid(acc_name)
                    local_target = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
                    subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_v2_latest.npz", local_target], capture_output=True, timeout=20)
                    git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_v2_latest.npz", f"chore(checkpoint): backup before failover from {acc_name}")
                except Exception:
                    pass
                colab_pool.mark_account_exhausted(acc_name, hours=12)
                return "FAILOVER"
            continue
        else:
            api_dead_count = 0

        # 2. Check training log via HTTP REST API (takes 0.2s, immune to GPU 100% load)
        ok, log_content = fetch_remote_train_log(acc_name)

        if ok and log_content:
            consecutive_log_fails = 0
            raw_lines = [l.strip() for l in log_content.splitlines() if l.strip()]

            # Reset cursor if log was cleared/restarted
            if len(raw_lines) < last_printed_line_idx:
                last_printed_line_idx = 0

            if last_printed_line_idx == 0:
                # First cycle: display the last 3 lines for immediate context
                initial_slice = raw_lines[-3:] if len(raw_lines) >= 3 else raw_lines
                for line in initial_slice:
                    print(f"[{get_vn_time_str()} | {acc_name}] {line}", flush=True)
                last_printed_line_idx = len(raw_lines)
                last_progress_time = time.time()
            elif len(raw_lines) > last_printed_line_idx:
                # Strictly stream ONLY newly appended lines (never re-print old lines)
                new_lines = raw_lines[last_printed_line_idx:]
                for line in new_lines:
                    print(f"[{get_vn_time_str()} | {acc_name}] {line}", flush=True)
                last_printed_line_idx = len(raw_lines)
                last_progress_time = time.time()

            # Periodically download latest checkpoint and commit to git every 5 minutes (~5 iters)
            if cycle % 5 == 0:
                try:
                    ensure_session_valid(acc_name)
                    local_target = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_v2_latest.npz")
                    dl = subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_v2_latest.npz", local_target], capture_output=True, text=True)
                    if dl.returncode != 0:
                        dl = subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/apollo_stage2_v2_latest.npz", local_target], capture_output=True, text=True)
                    if dl.returncode == 0 and os.path.exists(local_target) and os.path.getsize(local_target) > 100_000:
                        git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_v2_latest.npz", f"chore(checkpoint): update stage 2 checkpoint [{acc_name}]")
                except Exception as e:
                    print(f"[RELAY SYNC WARNING] Lỗi đồng bộ checkpoint: {e}", flush=True)

            if "STAGE 2 v2 TRAINING COMPLETE!" in log_content:
                print("\n" + "=" * 64)
                print("  🎉🎉🎉 HUẤN LUYỆN HOÀN TẤT 100%! CÁN ĐÍCH 150M BƯỚC! 🎉🎉🎉")
                print("=" * 64, flush=True)
                final_local = os.path.join(LOCAL_CKPT_DIR, "apollo_stage2_final.npz")
                ensure_session_valid(acc_name)
                subprocess.run(["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_final.npz", final_local])
                git_commit_and_push("colab_output/checkpoints_stage2/apollo_stage2_final.npz", "feat(weights): save final Apollo Stage 2 trained model (150M steps)")
                return "COMPLETE"
        elif log_content == "AUTH_EXPIRED":
            consecutive_log_fails += 1
            print(f"[{get_vn_time_str()} | {acc_name}] [AUTH RECOVERY] Token hết hạn (401), tự động refresh OAuth token và tái kết nối...", flush=True)
            colab_pool.refresh_account_token(acc_name)
            state._client = None
            state._auth_provider = None
            state._sessions = None
            ensure_session_valid(acc_name)
            continue
        elif log_content == "SESSION_NOT_FOUND":
            consecutive_log_fails += 1
            print(f"[{get_vn_time_str()} | {acc_name}] [SESSION RECOVERY] Không tìm thấy phiên local (#{consecutive_log_fails}/5)! Đang tự động khôi phục...", flush=True)
            colab_pool.refresh_account_token(acc_name)
            state._client = None
            state._auth_provider = None
            state._sessions = None
            healed = ensure_session_valid(acc_name)
            if not healed:
                is_alive, _ = is_vm_assigned_on_google(acc_name=acc_name)
                if not is_alive and consecutive_log_fails >= 3:
                    print(f"[RELAY FAILOVER] Máy ảo trên {acc_name} đã mất thực sự trên Google!")
                    colab_pool.mark_account_exhausted(acc_name, hours=12)
                    return "FAILOVER"
            continue
        elif log_content == "FILE_NOT_FOUND":
            elapsed_from_start = time.time() - launch_time
            if elapsed_from_start < 180:
                print(f"[{get_vn_time_str()} | {acc_name}] [INIT] Đang khởi tạo mô hình / JIT compile JAX trên GPU ({int(elapsed_from_start)}s)...", flush=True)
            else:
                print(f"[{get_vn_time_str()} | {acc_name}] [CẢNH BÁO] train.log chưa xuất hiện sau {int(elapsed_from_start)}s! Kiểm tra trạng thái tiến trình...", flush=True)
                if not check_is_training_running(acc_name):
                    print(f"[{get_vn_time_str()} | {acc_name}] [WARNING] Tiến trình train đã dừng trên máy ảo! Tự động khởi động lại...", flush=True)
                    deploy_and_start_training(acc_name, is_new=False)
                    last_progress_time = time.time()
                    launch_time = time.time()
                    last_printed_line_idx = 0
        else:
            consecutive_log_fails += 1
            print(f"[{get_vn_time_str()} | {acc_name}] [LOG REST WARNING] Không thể đọc log qua HTTP API (#{consecutive_log_fails}/5): {log_content}", flush=True)
            if consecutive_log_fails >= 5:
                is_alive, _ = is_vm_assigned_on_google(acc_name=acc_name)
                if not is_alive:
                    print(f"[RELAY FAILOVER] Quá 5 lần lỗi log và máy ảo không còn trên Google Colab!")
                    colab_pool.mark_account_exhausted(acc_name, hours=12)
                    return "FAILOVER"

        # 3. Check process health only if log has stalled for >10 minutes (600 seconds)
        if time.time() - last_progress_time > 600:
            print(f"[{get_vn_time_str()} | {acc_name}] [HEALTH CHECK] Log chưa cập nhật sau 10 phút, kiểm tra tiến trình python...", flush=True)
            if not check_is_training_running(acc_name):
                print(f"[{get_vn_time_str()} | {acc_name}] [WARNING] Tiến trình train đã dừng trên máy ảo! Tự động khởi động lại từ checkpoint gần nhất...", flush=True)
                deploy_and_start_training(acc_name, is_new=False)
                last_progress_time = time.time()
                launch_time = time.time()
                last_printed_line_idx = 0
            else:
                last_progress_time = time.time()  # Process is running, reset timer


def run_relay():
    print("=" * 64)
    print("  HỆ THỐNG ĐIỀU PHỐI XOAY VÒNG TÀI KHOẢN (Colab Relay Orchestrator)")
    print("=" * 64, flush=True)

    # First check if an account is already actively running training on a real GPU
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
            print(f"[RELAY] Huấn luyện GPU đang chạy sẵn trên {acc}, gắn trực tiếp vào giám sát!", flush=True)
            success = True
        elif has_assignment and not is_running:
            print(f"[RELAY] Máy ảo GPU {acc} đã có sẵn nhưng tiến trình đã dừng, khởi chạy lại...", flush=True)
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
            pull_git_latest()
            time.sleep(5)
            continue


if __name__ == "__main__":
    run_relay()
