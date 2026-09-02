"""
Download ONLY .npz checkpoint files from Kaggle kernel output using REST API.
Skips git repos and large auxiliary files.
"""
import os, sys, json, base64, urllib.request, urllib.parse, zipfile, io, re

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_json = os.path.join(root, "gpu", "kaggle.json")
    with open(kaggle_json) as f:
        creds = json.load(f)
    username = creds["username"]
    api_key  = creds["key"]
    kernel_slug = f"{username}/apollo-humanoid-mjx-training-dual-t4"

    out_dir = os.path.join(root, "kaggle_output", "checkpoints_v5")
    os.makedirs(out_dir, exist_ok=True)

    auth = base64.b64encode(f"{username}:{api_key}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "User-Agent": "kaggle/1.6"}

    # Step 1: Get list of output files via Kaggle REST API
    list_url = f"https://www.kaggle.com/api/v1/kernels/{kernel_slug}/output?page_size=100"
    print(f"Fetching file list from: {list_url}")
    req = urllib.request.Request(list_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"API list error: {e}")
        data = {}

    files = data.get("files", [])
    print(f"Total output files found: {len(files)}")

    npz_files = [f for f in files if f.get("name", "").endswith(".npz")]
    print(f"NPZ checkpoint files: {len(npz_files)}")
    for f in npz_files:
        print(f"  {f['name']} ({f.get('size', '?')} bytes)")

    # Step 2: Download each .npz via direct URL
    downloaded = []
    for finfo in npz_files:
        fname = finfo["name"]
        # Construct download URL (Kaggle REST v1)
        fname_enc = urllib.parse.quote(fname, safe="")
        dl_url = f"https://www.kaggle.com/api/v1/kernels/{kernel_slug}/output?fileName={fname_enc}"
        out_path = os.path.join(out_dir, os.path.basename(fname))

        print(f"\nDownloading: {fname} -> {out_path}")
        dl_req = urllib.request.Request(dl_url, headers=headers)
        try:
            with urllib.request.urlopen(dl_req, timeout=120) as resp:
                data_bytes = resp.read()
            with open(out_path, "wb") as wf:
                wf.write(data_bytes)
            print(f"  OK: {len(data_bytes):,} bytes")
            downloaded.append(out_path)
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")

    # Step 3: If REST API gave no files, fall back to CLI but pipe output
    if not downloaded:
        print("\nREST API returned no files, trying CLI download (npz only)...")
        import subprocess
        result = subprocess.run(
            ["kaggle", "kernels", "output", kernel_slug, "-p", out_dir],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr[-1000:])

    # Final list
    print("\n=== Checkpoint files available ===")
    all_npz = sorted([f for f in os.listdir(out_dir) if f.endswith(".npz") and os.path.getsize(os.path.join(out_dir, f)) > 100_000],
                     key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
                     reverse=True)

    if all_npz:
        for fn in all_npz:
            sz = os.path.getsize(os.path.join(out_dir, fn))
            print(f"  {fn}  ({sz:,} bytes)")
        best = os.path.join(out_dir, all_npz[0])
        print(f"\nBest checkpoint: {best}")
        return best
    else:
        # Fall back to existing v2 checkpoints
        v2_dir = os.path.join(root, "kaggle_output", "checkpoints")
        v2_npz = sorted([f for f in os.listdir(v2_dir) if f.endswith(".npz") and os.path.getsize(os.path.join(v2_dir, f)) > 100_000],
                        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
                        reverse=True)
        if v2_npz:
            best = os.path.join(v2_dir, v2_npz[0])
            print(f"\nNo v5 checkpoints, using best existing: {best}")
            return best
        print("ERROR: No valid checkpoint found!")
        return None

if __name__ == "__main__":
    main()
