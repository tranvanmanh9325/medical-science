"""
Colab Cloud Keeper for GitHub Actions
Runs on GitHub Actions to keep the Google Colab session alive while the user sleeps.
"""

import subprocess
import time
import sys
import os

SESSION_NAME = "stage2-train"


def main():
    print("=" * 64)
    print("  COLAB CLOUD KEEPER (GitHub Actions Cloud Runner)")
    print(f"  Target Session: {SESSION_NAME}")
    print("=" * 64, flush=True)

    # 1. Verify session exists
    res = subprocess.run(["colab", "sessions"], capture_output=True, text=True)
    print(res.stdout)
    if SESSION_NAME not in res.stdout:
        print(f"[ERROR] Session '{SESSION_NAME}' not found in colab sessions!")
        sys.exit(1)

    # 2. Keep-alive loop
    max_duration_seconds = 5.5 * 3600  # 5.5 hours (within GitHub Actions 6h limit)
    start_time = time.time()
    iter_count = 0

    while time.time() - start_time < max_duration_seconds:
        iter_count += 1
        elapsed_min = (time.time() - start_time) / 60.0

        try:
            check_code = """
with open('/content/train.log', 'r') as f:
    lines = f.readlines()
    print('LOG_TAIL:' + (lines[-1].strip() if lines else 'EMPTY'))
"""
            cmd = ["colab", "exec", "-s", SESSION_NAME]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = p.communicate(input=check_code, timeout=30)

            log_line = ""
            for line in out.splitlines():
                if "LOG_TAIL:" in line:
                    log_line = line.replace("LOG_TAIL:", "").strip()

            print(f"[{time.strftime('%H:%M:%S')}] Heartbeat #{iter_count} ({elapsed_min:.1f}m): {log_line}", flush=True)

            if "STAGE 2 v2 TRAINING COMPLETE!" in out:
                print("\n" + "=" * 64)
                print("  🎉 TRAINING COMPLETED SUCCESSFULLY ON COLAB! 🎉")
                print("=" * 64, flush=True)
                os.makedirs("artifacts", exist_ok=True)
                subprocess.run(
                    ["colab", "download", "-s", SESSION_NAME, "/content/checkpoints/apollo_stage2_final.npz", "artifacts/apollo_stage2_final.npz"]
                )
                print("[KEEPER] Final model downloaded to artifacts/apollo_stage2_final.npz", flush=True)
                break

        except Exception as e:
            print(f"[KEEPER WARNING] Heartbeat exception: {e}", flush=True)

        time.sleep(60)

    print("[KEEPER] Finished cloud keeper execution.", flush=True)


if __name__ == "__main__":
    main()
