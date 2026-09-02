import os
import json
import shutil
import stat

def remove_readonly(func, path, excinfo):
    """Clear readonly bit and retry deletion on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def prepare_kaggle_kernel():
    """
    Prepares the Kaggle Kernel package and metadata for GPU T4 x2 execution.
    Uses credentials at gpu/kaggle.json.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_json_path = os.path.join(root_dir, "gpu", "kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"Kaggle credentials not found at: {kaggle_json_path}")

    # Read username from kaggle.json
    with open(kaggle_json_path, 'r') as f:
        creds = json.load(f)
    username = creds.get('username', 'user')

    # Kaggle kernel output directory
    kernel_dir = os.path.join(root_dir, "kaggle_kernel_deploy")
    os.makedirs(kernel_dir, exist_ok=True)

    # 1. Generate kernel-metadata.json configured for GPU T4 x2
    kernel_slug = "apollo-humanoid-mjx-training"
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "Apollo Humanoid MJX Training - Dual T4 GPU",
        "code_file": "kaggle_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[METADATA GENERATED] {metadata_path}")

    # 2. Copy source codes to deployment folder
    shutil.copy(os.path.join(root_dir, "training", "kaggle_train.py"), os.path.join(kernel_dir, "kaggle_train.py"))
    
    training_pkg_dir = os.path.join(kernel_dir, "training")
    os.makedirs(training_pkg_dir, exist_ok=True)
    shutil.copy(os.path.join(root_dir, "training", "env_apollo_mjx.py"), os.path.join(training_pkg_dir, "env_apollo_mjx.py"))
    shutil.copy(os.path.join(root_dir, "training", "rewards.py"), os.path.join(training_pkg_dir, "rewards.py"))
    shutil.copy(os.path.join(root_dir, "training", "ppo_mjx_trainer.py"), os.path.join(training_pkg_dir, "ppo_mjx_trainer.py"))

    # 3. Copy DeepMind Apollo model files (ignoring .git)
    src_apollo = os.path.join(root_dir, "google_deepmind_menagerie", "apptronik_apollo")
    dst_apollo = os.path.join(kernel_dir, "google_deepmind_menagerie", "apptronik_apollo")
    
    if os.path.exists(dst_apollo):
        shutil.rmtree(dst_apollo, onerror=remove_readonly)
    
    os.makedirs(os.path.dirname(dst_apollo), exist_ok=True)
    shutil.copytree(src_apollo, dst_apollo)

    print(f"[PACKAGE COMPLETE] Kernel ready for deployment at: {kernel_dir}")
    print("Run `kaggle kernels push -p ./kaggle_kernel_deploy` when ready to start training.")

if __name__ == "__main__":
    prepare_kaggle_kernel()
