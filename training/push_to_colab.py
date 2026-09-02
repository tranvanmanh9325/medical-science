"""
================================================================================
 Apptronik Apollo Humanoid - Google Colab Training Package & Pipeline Sync
--------------------------------------------------------------------------------
 Tự động chuẩn bị, đóng gói và đồng bộ hóa mã nguồn huấn luyện lên Google Colab
 (Tương thích 100% với quy trình Kaggle: JAX + MuJoCo MJX 4096 Envs PPO)
================================================================================
"""

import os
import sys
import json
import shutil
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def prepare_colab_package():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    colab_deploy_dir = os.path.join(root_dir, "colab_deploy")
    os.makedirs(colab_deploy_dir, exist_ok=True)

    print("=" * 64)
    print("  [GOOGLE COLAB PIPELINE SYNC] ĐỒNG BỘ MÔI TRƯỜNG HUẤN LUYỆN")
    print("=" * 64)

    # 1. Đồng bộ file Notebook chính
    src_nb = os.path.join(root_dir, "kaggle_kernel_deploy", "apollo_humanoid_mjx_training.ipynb")
    dst_root_nb = os.path.join(root_dir, "colab_apollo_training.ipynb")
    dst_colab_nb = os.path.join(colab_deploy_dir, "colab_apollo_humanoid_mjx_training.ipynb")

    if os.path.exists(src_nb):
        shutil.copyfile(src_nb, dst_root_nb)
        shutil.copyfile(src_nb, dst_colab_nb)
        print(f"[1/3] Đã đồng bộ Notebook: {dst_root_nb}")
    else:
        print(f"[WARNING] Không tìm thấy file gốc: {src_nb}")

    # 2. Kiểm tra tài nguyên mô hình 3D Apollo
    model_dir = os.path.join(root_dir, "google_deepmind_menagerie", "apptronik_apollo")
    scene_path = os.path.join(model_dir, "scene.xml")
    if os.path.exists(scene_path):
        print(f"[2/3] Mô hình 3D Robot Apollo: SẴN SÀNG ({scene_path})")
    else:
        print(f"[ERROR] Thiếu file mô hình: {scene_path}")

    # 3. Tự động Commit & Push lên GitHub
    print("[3/3] Đang đồng bộ lên GitHub repo...")
    try:
        subprocess.run(["git", "add", "colab_apollo_training.ipynb", "colab_deploy/"], cwd=root_dir, check=True)
        res = subprocess.run(["git", "status", "--porcelain"], cwd=root_dir, capture_output=True, text=True)
        if res.stdout.strip():
            subprocess.run(["git", "commit", "-m", "sync(colab): automatic pipeline synchronization"], cwd=root_dir, check=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=root_dir, check=True)
            print("--> Đã đẩy bản cập nhật mới nhất lên GitHub thành công!")
        else:
            print("--> Mã nguồn trên GitHub đã ở trạng thái mới nhất!")
    except Exception as e:
        print(f"[GIT INFO] {e}")

    colab_url = "https://colab.research.google.com/github/tranvanmanh9325/medical-science/blob/main/colab_apollo_training.ipynb"
    print("\n" + "=" * 64)
    print("  [SẴN SÀNG HUẤN LUYỆN GOOGLE COLAB GPU]")
    print(f"  Link 1-Click: {colab_url}")
    print("  Lệnh CLI chạy ngầm: colab run --gpu T4 training/colab_train.py")
    print("=" * 64)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--run", "-r", "--cli"):
        print("[COLAB CLI] Khởi động phiên huấn luyện GPU T4 chạy ngầm từ xa...")
        os.system("colab run --gpu T4 training/colab_train.py")
    else:
        prepare_colab_package()

