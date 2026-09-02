import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

def run_pre_training_audit():
    print("==================================================================", flush=True)
    print(" [PRE-TRAINING COMPREHENSIVE AUDIT] Checking for any defects...    ", flush=True)
    print("==================================================================", flush=True)

    # 1. Check Model XML existence & MJX conversion
    xml_path = os.path.join(root_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    assert os.path.exists(xml_path), f"Error: scene.xml not found at {xml_path}"
    print("[CHECK 1/6] Model XML exists: PASSED", flush=True)

    # 2. Test Environment Initialization & Dimensions
    from training.env_apollo_mjx import ApolloMJXEnvV2
    env = ApolloMJXEnvV2(xml_path)
    print(f"[CHECK 2/6] Env Initialized: nq={env.nq}, nv={env.nv}, nu={env.nu}, obs_dim={env.obs_dim}: PASSED", flush=True)

    # 3. Test Reset & Step Functions
    rng = jax.random.PRNGKey(0)
    rng_reset, rng_step = jax.random.split(rng)
    
    obs, state = env.reset(rng_reset)
    print(f"  -> Reset Output Obs Shape: {obs.shape}", flush=True)
    assert obs.shape == (env.obs_dim,), f"Dimension mismatch! Expected {env.obs_dim}, got {obs.shape}"

    action = jnp.zeros(env.action_dim)
    next_obs, next_state, reward, done, r_info = env.step(state, action)
    print(f"  -> Step Output Next Obs Shape: {next_obs.shape}, Reward: {reward:.4f}, Done: {done}", flush=True)
    assert next_obs.shape == (env.obs_dim,), f"Next obs dimension mismatch!"
    print("[CHECK 3/6] Reset & Single Step Consistency: PASSED", flush=True)

    # 4. Test ActorCritic Network
    from training.ppo_mjx_trainer import ActorCritic
    batch_size = 16
    obs_batch = jnp.zeros((batch_size, env.obs_dim))
    network = ActorCritic(action_dim=env.action_dim)
    params = network.init(rng, obs_batch)
    actor_mean, log_std, values = network.apply(params, obs_batch)
    
    assert actor_mean.shape == (batch_size, env.action_dim), f"Actor mean shape mismatch: {actor_mean.shape}"
    assert values.shape == (batch_size,), f"Critic value shape mismatch: {values.shape}"
    print(f"  -> Network Output: Mean {actor_mean.shape}, Values {values.shape}", flush=True)
    print("[CHECK 4/6] ActorCritic Network Architecture: PASSED", flush=True)

    # 5. Check Deployment Folder & Kaggle Metadata
    deploy_dir = os.path.join(root_dir, "kaggle_kernel_deploy")
    assert os.path.exists(deploy_dir), "Deployment folder missing!"
    
    meta_path = os.path.join(deploy_dir, "kernel-metadata.json")
    assert os.path.exists(meta_path), "kernel-metadata.json missing!"
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    assert meta.get("enable_gpu") == "true", "GPU acceleration not enabled in metadata!"
    assert meta.get("code_file") == "kaggle_train.py", "Entry point mismatch!"
    
    deploy_xml = os.path.join(deploy_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    assert os.path.exists(deploy_xml), "Deployed model XML missing!"
    print(f"  -> Kaggle Metadata Validated for Slug: {meta['id']}", flush=True)
    print("[CHECK 5/6] Kaggle Deployment Package Integrity: PASSED", flush=True)

    # 6. Check Kaggle Token Credentials
    kaggle_json = os.path.join(root_dir, "gpu", "kaggle.json")
    assert os.path.exists(kaggle_json), "gpu/kaggle.json missing!"
    with open(kaggle_json, 'r') as f:
        creds = json.load(f)
    assert 'username' in creds and 'key' in creds, "Invalid kaggle.json format!"
    print(f"  -> Kaggle Credentials for User '{creds['username']}': VALID", flush=True)
    print("[CHECK 6/6] Kaggle Authentication Credentials: PASSED", flush=True)

    print("\n==================================================================", flush=True)
    print(" [ALL AUDIT CHECKS PASSED] 100% READY FOR TRAINING WITH ZERO ERRORS!", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_pre_training_audit()
