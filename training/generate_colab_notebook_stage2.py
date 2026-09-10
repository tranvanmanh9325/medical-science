import os
import json


def generate():
    root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = os.path.join(root_dir, "colab_deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    train_script_path = os.path.join(deploy_dir, "train_stage2.py")
    with open(train_script_path, "r", encoding="utf-8") as f:
        training_code = f.read()

    # Adapt CLI parser for Jupyter notebook environment
    training_code_adapted = training_code.replace(
        "args = parser.parse_args()",
        "class _Args: resume = ''\nargs = _Args()"
    )

    INSTALL_CELL = [
        "# Step 1: Check GPU (Runtime > Change runtime type > T4 / A100 GPU)\n",
        "!nvidia-smi\n",
        "!pip install -q --no-cache-dir mujoco mujoco-mjx flax optax\n",
        "import jax\n",
        "print('Backend:', jax.default_backend(), '| Devices:', jax.devices())\n",
        "assert jax.default_backend() in ('gpu','tpu'), 'Enable GPU in Runtime > Change runtime type!'"
    ]

    DRIVE_CELL = [
        "# Step 2: Set GITHUB_TOKEN (Optional - for automated checkpoint push)\n",
        "import os\n",
        "GITHUB_TOKEN = ''  # <-- Paste your GitHub token here for auto-push to repo\n",
        "if GITHUB_TOKEN:\n",
        "    with open('/content/github_token.txt', 'w') as f:\n",
        "        f.write(GITHUB_TOKEN)\n",
        "    print('[OK] GitHub token configured')\n",
        "else:\n",
        "    print('[INFO] No GITHUB_TOKEN - checkpoints will be saved locally to /content/checkpoints/')"
    ]

    DOWNLOAD_CELL = [
        "# Step 3: Download Apollo robot model\n",
        "import os, urllib.request, zipfile, shutil\n",
        "TARGET    = '/content/mujoco_menagerie'\n",
        "APOLLO_XML = os.path.join(TARGET, 'apptronik_apollo', 'scene.xml')\n",
        "if not os.path.exists(APOLLO_XML):\n",
        "    print('Downloading mujoco_menagerie...')\n",
        "    url = 'https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip'\n",
        "    urllib.request.urlretrieve(url, '/tmp/men.zip')\n",
        "    with zipfile.ZipFile('/tmp/men.zip') as z: z.extractall('/tmp/men_ex')\n",
        "    shutil.move('/tmp/men_ex/mujoco_menagerie-main', TARGET)\n",
        "    os.remove('/tmp/men.zip')\n",
        "assert os.path.exists(APOLLO_XML)\n",
        "print(f'[OK] Apollo model ready: {APOLLO_XML}')"
    ]

    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": INSTALL_CELL},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": DRIVE_CELL},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": DOWNLOAD_CELL},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [training_code_adapted]},
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "name": "Apollo Humanoid Stage 2 Locomotion v9"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 0
    }

    nb_path = os.path.join(deploy_dir, "apollo_stage2_walking_colab.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[COLAB STAGE 2 v9] Generated {nb_path} ({os.path.getsize(nb_path) // 1024}KB)")
    return nb_path


if __name__ == "__main__":
    generate()
