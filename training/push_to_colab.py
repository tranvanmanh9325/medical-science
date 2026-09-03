"""
Push and run Apollo Humanoid training on Google Colab GPU via colab CLI.
Usage:
    python training/push_to_colab.py          # generate + push + run (default)
    python training/push_to_colab.py --status # check running sessions
    python training/push_to_colab.py --stop   # stop all apollo sessions
    python training/push_to_colab.py --log    # stream logs from active session
"""

import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(ROOT, "colab_deploy", "apollo_colab_train.py")
SESSION_NAME = "apollo-stage1"


def run_cmd(args, check=True, capture=False):
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        capture_output=capture,
        text=True
    )


def generate_train_script():
    """Generate self-contained Python training script for colab run."""
    os.makedirs(os.path.join(ROOT, "colab_deploy"), exist_ok=True)

    # Same v15 training code — adapted as standalone .py for `colab run`
    # Checkpoints saved to /root/checkpoints/ on the Colab VM
    # (use `colab download` to pull them back)
    code = r'''#!/usr/bin/env -S colab run --gpu T4
"""
Apollo Humanoid Stage 1 Standing Balance — v15
Runs headlessly on Colab GPU via: colab run --gpu T4 apollo_colab_train.py
"""
import os, sys, time, math, urllib.request, zipfile, shutil

# ── Download mujoco_menagerie ──────────────────────────────────────────────────
TARGET = "/root/mujoco_menagerie"
APOLLO_XML = os.path.join(TARGET, "apptronik_apollo", "scene.xml")
if not os.path.exists(APOLLO_XML):
    print("Downloading mujoco_menagerie...")
    url = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
    zip_path = "/tmp/menagerie.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/tmp/menagerie_extract")
    shutil.move("/tmp/menagerie_extract/mujoco_menagerie-main", TARGET)
    os.remove(zip_path)
    print("Done.")
assert os.path.exists(APOLLO_XML)
print("[OK] Apollo model:", APOLLO_XML)

# ── Install dependencies ───────────────────────────────────────────────────────
os.system("pip install -q --no-cache-dir mujoco mujoco-mjx flax optax")

import jax, jax.numpy as jnp, optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 1 STANDING BALANCE (v15 Colab)")
print("=" * 64)
print("Backend:", jax.default_backend(), "| Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "No GPU found!"

CKPT_DIR = "/root/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
mj_model = mujoco.MjModel.from_xml_path(APOLLO_XML)
SIM_DT, N_SUBSTEPS = 0.002, 5
CTRL_DT = SIM_DT * N_SUBSTEPS

mj_model.opt.timestep      = SIM_DT
mj_model.opt.iterations    = 4
mj_model.opt.ls_iterations = 4
mj_model.opt.integrator    = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
for i in range(mj_model.ngeom):
    mj_model.geom_solref[i, 0] = 0.004
    mj_model.geom_solref[i, 1] = 1.0
    mj_model.geom_solimp[i, :] = [0.9, 0.95, 0.001, 0.5, 2.0]

mjx_model = mjx.put_model(mj_model)
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
ctrl_range  = jnp.array(mj_model.actuator_ctrlrange)

key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id < 0: key_id = 0
default_qpos = jnp.array(mj_model.key_qpos[key_id])
default_ctrl = jnp.array(mj_model.key_qpos[key_id][7:])  # position actuator target
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL    = float(default_qpos[2])
ACTION_SCALE = 0.1
EPISODE_LEN  = 1000
TERM_HEIGHT  = Z_NOMINAL * 0.75
TERM_TILT    = 0.5

OBS_DIM = 3 + 3 + 3 + nu + nu + nu

class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in (512, 256, 128):
            x = nn.elu(nn.Dense(h)(x))
        mean    = nn.tanh(nn.Dense(self.action_dim)(x))
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        log_std = jnp.clip(log_std, -3.0, 0.5)
        return mean, log_std, nn.Dense(1)(x).squeeze(-1)

def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([2.*(qx*qz+qw*qy), 2.*(qy*qz-qw*qx), 1.-2.*(qx**2+qy**2)])

def get_obs(d, prev_act):
    upvec  = get_upvector(d.qpos)
    jpos   = d.qpos[7:7+nu] - default_pose
    return jnp.clip(jnp.concatenate([upvec, d.qvel[:3], d.qvel[3:6], jpos, d.qvel[6:6+nu], prev_act]), -20., 20.)

def compute_reward(qpos, qvel, action, prev_action):
    upvec = get_upvector(qpos)
    r_ori    = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.05)
    r_height = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.04)
    r_lin    = jnp.exp(-jnp.sum(jnp.square(qvel[:2])) / 0.5)
    r_ang    = jnp.exp(-jnp.sum(jnp.square(qvel[3:6])) / 0.5)
    r_pose   = jnp.exp(-jnp.mean(jnp.square(qpos[7:7+nu] - default_pose)) / 0.1)
    p_smooth = 0.02 * jnp.mean(jnp.square(action - prev_action))
    return jnp.maximum(0., r_ori + r_height + r_lin + r_ang + r_pose + 0.2 - p_smooth) * CTRL_DT

def env_reset(rng):
    rng_q, rng_v, rng_j = jax.random.split(rng, 3)
    noise = jax.random.uniform(rng_j, (nq-7,), minval=-0.05, maxval=0.05)
    qpos  = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.01, maxval=0.01),
        default_qpos[7:] + noise
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.05, maxval=0.05)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)
    return {"d": d, "prev_act": jnp.zeros(nu), "step": jnp.zeros((), jnp.int32)}

def env_step(state, action_and_rng):
    raw_act, rng_reset = action_and_rng
    d, prev_act, step  = state["d"], state["prev_act"], state["step"]
    ctrl = jnp.clip(default_ctrl + raw_act * ACTION_SCALE, ctrl_range[:,0], ctrl_range[:,1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)
    rew      = compute_reward(d.qpos, d.qvel, raw_act, prev_act)
    upvec    = get_upvector(d.qpos)
    done     = jnp.logical_or(
        jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT),
        step + 1 >= EPISODE_LEN
    )
    rst      = env_reset(rng_reset)
    next_d   = jax.tree.map(lambda r, c: jnp.where(done, r, c), rst["d"], d)
    return (get_obs(d, raw_act),
            {"d": next_d, "prev_act": jnp.where(done, jnp.zeros(nu), raw_act),
             "step": jnp.where(done, jnp.zeros((), jnp.int32), step+1)},
            rew, upvec[2] < TERM_TILT, step+1 >= EPISODE_LEN)

# ── PPO ────────────────────────────────────────────────────────────────────────
NUM_ENVS, ROLLOUT = 4096, 32
GAMMA, LAM, CLIP_EPS = 0.99, 0.95, 0.2
ENT_COEF, VF_COEF, MAX_GRAD = 0.01, 0.5, 0.5
TOTAL_STEPS  = 100_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT

network    = ActorCritic(action_dim=nu)
rng        = jax.random.PRNGKey(42)
rng, ri    = jax.random.split(rng)
params     = network.init(ri, jnp.zeros((1, OBS_DIM)))
tx         = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(optax.linear_schedule(3e-4, 3e-5, N_ITERS), eps=1e-5))
opt_state  = tx.init(params)
states     = jax.vmap(env_reset)(jax.random.split(rng, NUM_ENVS))

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, rr = jax.random.split(r, 3)
        obs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(st)
        mu, ls, val = network.apply(p, obs)
        std = jnp.exp(ls)
        act = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp  = jnp.clip(-0.5 * jnp.sum(jnp.square((act-mu)/(std+1e-8)) +
                        2.*ls + math.log(2.*math.pi), axis=-1), -10., 10.)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, jax.random.split(rr, NUM_ENVS)))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(_step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj
    _, _, nv_last = network.apply(params, jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(fst))

    def _gae(carry, t):
        gae, nxv = carry
        done  = jnp.logical_or(terms[t], truncs[t])
        delta = rews[t] + GAMMA * nxv * (1.-terms[t].astype(jnp.float32)) - vals[t]
        gae   = delta + GAMMA * LAM * (1.-done.astype(jnp.float32)) * gae
        return (gae, vals[t]), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last), jnp.arange(ROLLOUT-1,-1,-1))
    advs = jnp.flip(advs, 0); rets = advs + vals
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret, ovf = *map(flat, [obs, act, old_lp, advs, rets]), flat(vals)

    def loss_fn(p):
        mu, ls, v = network.apply(p, fo)
        std = jnp.exp(ls)
        lp  = jnp.clip(-0.5*jnp.sum(jnp.square((fa-mu)/(std+1e-8))+2.*ls+math.log(2.*math.pi),axis=-1),-10.,10.)
        ratio = jnp.exp(jnp.clip(lp-flp, -5., 5.))
        pg  = -jnp.mean(jnp.minimum(ratio*fadv, jnp.clip(ratio,1-CLIP_EPS,1+CLIP_EPS)*fadv))
        vc  = ovf + jnp.clip(v-ovf, -5., 5.)
        vf  = VF_COEF * jnp.mean(jnp.maximum(jnp.square(v-fret), jnp.square(vc-fret)))
        ent = -ENT_COEF * jnp.mean(jnp.sum(ls+0.5*math.log(2*math.pi*math.e), axis=-1))
        return jnp.where(jnp.isnan(pg+vf+ent), 0., pg+vf+ent)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0., g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, fst, rng, jnp.mean(rews), loss

# ── Training loop ──────────────────────────────────────────────────────────────
print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | CKPT_DIR={CKPT_DIR}")
t0, cur = time.time(), 0

for it in range(1, N_ITERS + 1):
    t1 = time.time()
    params, opt_state, states, rng, mr, loss = ppo_iter(params, opt_state, states, rng)
    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time()-t1)
    if it % 10 == 0 or it == 1:
        r = float(mr)
        status = "*** STANDING ***" if r>0.020 else "improving" if r>0.010 else "starting" if r>0.003 else "..."
        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | rew={r:.5f} | loss={float(loss):.4f} | sps={sps:,.0f} | t={time.time()-t0:.0f}s {status}", flush=True)
    if it % 50 == 0 or it == N_ITERS:
        ck = f"{CKPT_DIR}/apollo_stage1_v15_step_{cur}.npz"
        jnp.savez(ck, **flax.traverse_util.flatten_dict(params, sep="/"))
        print(f"  -> checkpoint: {ck}", flush=True)

print("TRAINING COMPLETE")
'''

    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[OK] Generated: {SCRIPT_PATH}")


def push_and_run():
    generate_train_script()
    print("\n[COLAB] Khởi động phiên GPU T4 và train ngầm...")
    print("        (Lần đầu cần `colab auth login` nếu chưa xác thực)\n")
    # colab run tự: cấp VM → upload script → chạy → giải phóng VM
    # --keep giữ VM để download checkpoint sau
    run_cmd([
        "colab", "run",
        "--gpu", "T4",
        "--session", SESSION_NAME,
        "--keep",          # giữ VM sau train để download checkpoint
        SCRIPT_PATH
    ])


def check_status():
    run_cmd(["colab", "sessions"])


def download_checkpoints():
    local_dir = os.path.join(ROOT, "kaggle_output", "colab_checkpoints")
    os.makedirs(local_dir, exist_ok=True)
    print(f"[COLAB] Downloading checkpoints → {local_dir}")
    # List và download từng file .npz
    result = run_cmd(["colab", "ls", f"--session={SESSION_NAME}", "/root/checkpoints/"],
                     capture=True, check=False)
    print(result.stdout)
    for line in result.stdout.strip().splitlines():
        fname = line.strip()
        if fname.endswith(".npz"):
            remote = f"/root/checkpoints/{fname}"
            local  = os.path.join(local_dir, fname)
            run_cmd(["colab", "download", f"--session={SESSION_NAME}", remote, local])
            print(f"  -> {local}")


def stop_session():
    run_cmd(["colab", "stop", f"--session={SESSION_NAME}"], check=False)
    print("[COLAB] Session stopped.")


def stream_log():
    print("[COLAB] Connecting to session log (Ctrl+C to detach)...")
    run_cmd(["colab", "log", f"--session={SESSION_NAME}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apollo Colab Training CLI")
    parser.add_argument("--status",   action="store_true", help="List active sessions")
    parser.add_argument("--stop",     action="store_true", help="Stop apollo session")
    parser.add_argument("--download", action="store_true", help="Download checkpoints from VM")
    parser.add_argument("--log",      action="store_true", help="Stream live log from session")
    parser.add_argument("--generate", action="store_true", help="Only generate script, don't push")
    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.stop:
        stop_session()
    elif args.download:
        download_checkpoints()
    elif args.log:
        stream_log()
    elif args.generate:
        generate_train_script()
    else:
        push_and_run()
