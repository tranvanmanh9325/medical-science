import os
import sys
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
TOKEN_FILE = os.path.join(CONFIG_DIR, "token.json")

os.makedirs(ACCOUNTS_DIR, exist_ok=True)

def add_account(acc_name):
    print("=" * 64)
    print(f"  THÊM TÀI KHOẢN GOOGLE MỚI VÀO POOL: {acc_name}")
    print("=" * 64)
    print("1. Trình duyệt web sẽ mở ra.")
    print("2. Vui lòng CHỌN TÀI KHOẢN GMAIL MỚI và bấm Cho phép (Allow).")
    print("=" * 64, flush=True)

    # 1. Clear current token temporarily
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

    # 2. Trigger colab login
    try:
        subprocess.run(["colab", "sessions"], check=False)
    except Exception as e:
        print(f"Error during auth: {e}")

    # 3. Check if new token was created
    if os.path.exists(TOKEN_FILE):
        dest = os.path.join(ACCOUNTS_DIR, f"{acc_name}.json")
        shutil.copy2(TOKEN_FILE, dest)
        print("\n" + "=" * 64)
        print(f"  🎉 ĐÃ LƯU THÀNH CÔNG TÀI KHOẢN: {acc_name}!")
        print(f"  File lưu tại: {dest}")
        print("=" * 64 + "\n", flush=True)
    else:
        print("\n[LỖI] Chưa nhận được token mới. Vui lòng thử lại.\n", flush=True)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "account_2"
    add_account(name)
