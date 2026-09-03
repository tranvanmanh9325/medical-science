import os
import json

def generate():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = os.path.join(root_dir, "colab_deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    # =========================================================
    # Colab notebook — same v15 training code as Kaggle
    # Differences vs Kaggle:
    #   - Mount Google Drive to save checkpoints persistently
    #   - JAX GPU auto-detected (T4/A100/V100 depending on plan)
    #   - Download mujoco_menagerie via urllib (no git needed)
    #   - Checkpoints saved to Google Drive
    # =========================================================

    TRAINING_CODE = r'''
import os, time, math
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 1 STANDING BALANCE (v15)")
print("  All-positive Gaussian reward, position actuator fix")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "GPU required! Enable in Runtime > Change runtime type"

# ================================================================
# 1. ACTOR-CRITIC
# ================================================================
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
        value   = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value

# ================================================================
# 2. PHYSICS MODEL
# ================================================================
model_path = "/content/mujoco_menagerie/apptronik_apollo/scene.xml"
mj_model   = mujoco.MjModel.from_xml_path(model_path)
SIM_DT, N_SUBSTEPS = 0.002, 5
CTRL_DT = SIM_DT * N_SUBSTEPS  # 0.01s

mj_model.opt.timestep      = SIM_DT
mj_model.opt.iterations    = 4
mj_model.opt.ls_iterations = 4
mj_model.opt.integrator    = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
for i in range(mj_model.ngeom):
    mj_model.geom_solref[i, 0] = 0.004
    mj_model.geom_solref[i, 1] = 1.0
    mj_model.geom_solimp[i, :] = [0.9, 0.95, 0.001, 0.5, 2.0]

mjx_model  = mjx.put_model(mj_model)
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
ctrl_range  = jnp.array(mj_model.actuator_ctrlrange)

key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id < 0: key_id = 0
default_qpos = jnp.array(mj_model.key_qpos[key_id])
# Apollo position actuator: ctrl = target joint position = key_qpos[7:]
default_ctrl = jnp.array(mj_model.key_qpos[key_id][7:])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL    = float(default_qpos[2])
ACTION_SCALE = 0.1
EPISODE_LEN  = 1000
TERM_HEIGHT  = Z_NOMINAL * 0.75
TERM_TILT    = 0.5

print(f"z_nominal={Z_NOMINAL:.4f}m | ctrl_dt={CTRL_DT:.3f}s | action_scale={ACTION_SCALE}")
OBS_DIM = 3 + 3 + 3 + nu + nu + nu

def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([
        2.0*(qx*qz + qw*qy),
        2.0*(qy*qz - qw*qx),
        1.0 - 2.0*(qx**2 + qy**2),
    ])

def get_obs(d, prev_act):
    upvec  = get_upvector(d.qpos)
    linvel = d.qvel[:3]
    angvel = d.qvel[3:6]
    jpos   = d.qpos[7:7+nu] - default_pose
    jvel   = d.qvel[6:6+nu]
    return jnp.clip(jnp.concatenate([upvec, linvel, angvel, jpos, jvel, prev_act]), -20.0, 20.0)

# ================================================================
# 3. REWARD — all-positive Gaussian, max=0.052/step when standing
# ================================================================
def compute_reward(qpos, qvel, action, prev_action):
    upvec = get_upvector(qpos)
    r_orientation = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.05)
    r_height      = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.04)
    r_still_lin   = jnp.exp(-jnp.sum(jnp.square(qvel[:2])) / 0.5)
    r_still_ang   = jnp.exp(-jnp.sum(jnp.square(qvel[3:6])) / 0.5)
    jpos = qpos[7:7+nu]
    r_pose = jnp.exp(-jnp.mean(jnp.square(jpos - default_pose)) / 0.1)
    r_alive   = 0.2
    p_smooth  = 0.02 * jnp.mean(jnp.square(action - prev_action))
    total = (r_orientation + r_height + r_still_lin + r_still_ang + r_pose + r_alive - p_smooth)
    return jnp.maximum(0.0, total) * CTRL_DT

# ================================================================
# 4. ENVIRONMENT
# ================================================================
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
    d, prev_act, step = state["d"], state["prev_act"], state["step"]
    ctrl = jnp.clip(default_ctrl + raw_act * ACTION_SCALE, ctrl_range[:,0], ctrl_range[:,1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)
    rew     = compute_reward(d.qpos, d.qvel, raw_act, prev_act)
    obs_out = get_obs(d, raw_act)
    upvec      = get_upvector(d.qpos)
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)
    reset_state = env_reset(rng_reset)
    next_d    = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act  = jnp.where(done, jnp.zeros(nu), raw_act)
    next_step = jnp.where(done, jnp.zeros((), jnp.int32), step_new)
    return obs_out, {"d": next_d, "prev_act": next_act, "step": next_step}, rew, terminated, truncated

# ================================================================
# 5. PPO
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 32
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
TOTAL_STEPS  = 100_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT

lr_schedule = optax.linear_schedule(3e-4, 3e-5, N_ITERS)
network     = ActorCritic(action_dim=nu)
rng         = jax.random.PRNGKey(42)
rng, ri     = jax.random.split(rng)
params      = network.init(ri, jnp.zeros((1, OBS_DIM)))
tx          = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(lr_schedule, eps=1e-5))
opt_state   = tx.init(params)
states      = jax.vmap(env_reset)(jax.random.split(rng, NUM_ENVS))

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs  = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        act  = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp   = jnp.clip(-0.5 * jnp.sum(
            jnp.square((act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, r_resets))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(_step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj
    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(fst)
    _, _, nv_last = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        done  = jnp.logical_or(terms[t], truncs[t])
        delta = rews[t] + GAMMA * nxv * (1. - terms[t].astype(jnp.float32)) - vals[t]
        gae   = delta + GAMMA * LAM * (1. - done.astype(jnp.float32)) * gae
        return (gae, vals[t]), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last), jnp.arange(ROLLOUT-1, -1, -1))
    advs  = jnp.flip(advs, axis=0)
    rets  = advs + vals
    advs  = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret, ovf = *map(flat, [obs, act, old_lp, advs, rets]), flat(vals)

    def loss_fn(p):
        mu, ls, v = network.apply(p, fo)
        std = jnp.exp(ls)
        lp  = jnp.clip(-0.5 * jnp.sum(jnp.square((fa-mu)/(std+1e-8)) +
                        2.*ls + math.log(2.*math.pi), axis=-1), -10., 10.)
        ratio = jnp.exp(jnp.clip(lp - flp, -5., 5.))
        pg  = -jnp.mean(jnp.minimum(ratio*fadv, jnp.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)*fadv))
        vc  = ovf + jnp.clip(v - ovf, -5., 5.)
        vf  = VF_COEF * jnp.mean(jnp.maximum(jnp.square(v-fret), jnp.square(vc-fret)))
        ent = -ENT_COEF * jnp.mean(jnp.sum(ls + 0.5*math.log(2*math.pi*math.e), axis=-1))
        return jnp.where(jnp.isnan(pg+vf+ent), 0.0, pg+vf+ent)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0., g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, fst, rng, jnp.mean(rews), loss

# ================================================================
# 6. TRAINING LOOP — saves to Google Drive
# ================================================================
os.makedirs(CKPT_DIR, exist_ok=True)
t0, cur = time.time(), 0
print(f"Checkpoints -> {CKPT_DIR}")
print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | Target rew>0.020")

for it in range(1, N_ITERS + 1):
    t1 = time.time()
    params, opt_state, states, rng, mr, loss = ppo_iter(params, opt_state, states, rng)
    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)
    if it % 10 == 0 or it == 1:
        r_val = float(mr)
        status = ("*** STANDING ***" if r_val > 0.020 else
                  "improving" if r_val > 0.010 else
                  "starting"  if r_val > 0.003 else "...")
        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | rew={r_val:.5f} | "
              f"loss={float(loss):.4f} | sps={sps:,.0f} | t={time.time()-t0:.0f}s {status}",
              flush=True)
    if it % 50 == 0 or it == N_ITERS:
        ck = f"{CKPT_DIR}/apollo_stage1_v15_step_{cur}.npz"
        jnp.savez(ck, **flax.traverse_util.flatten_dict(params, sep="/"))
        print(f"  -> checkpoint saved: {ck}", flush=True)

print("STAGE 1 v15 COMPLETE")
'''

    DOWNLOAD_CELL = [
        "# Download mujoco_menagerie via ZIP (no git/auth needed)\n",
        "import os, urllib.request, zipfile, shutil\n",
        "\n",
        "TARGET    = '/content/mujoco_menagerie'\n",
        "APOLLO_XML = os.path.join(TARGET, 'apptronik_apollo', 'scene.xml')\n",
        "\n",
        "if not os.path.exists(APOLLO_XML):\n",
        "    print('Downloading mujoco_menagerie ZIP...')\n",
        "    zip_url  = 'https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip'\n",
        "    zip_path = '/tmp/mujoco_menagerie.zip'\n",
        "    urllib.request.urlretrieve(zip_url, zip_path)\n",
        "    print(f'Downloaded: {os.path.getsize(zip_path)/1e6:.1f} MB')\n",
        "    with zipfile.ZipFile(zip_path, 'r') as z:\n",
        "        z.extractall('/tmp/menagerie_extract')\n",
        "    shutil.move('/tmp/menagerie_extract/mujoco_menagerie-main', TARGET)\n",
        "    os.remove(zip_path)\n",
        "\n",
        "assert os.path.exists(APOLLO_XML), f'Missing: {APOLLO_XML}'\n",
        "print('[OK] Apollo model ready.')"
    ]

    cells = [
        # Cell 0: GPU check
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "# Check GPU — go to Runtime > Change runtime type > GPU\n",
             "!nvidia-smi\n",
             "import jax\n",
             "print('Devices:', jax.devices())"
         ]},
        # Cell 1: Install packages
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "!pip install -q --no-cache-dir mujoco mujoco-mjx flax optax\n",
             "import jax\n",
             "print('Backend:', jax.default_backend(), '| Devices:', jax.devices())\n",
             "assert jax.default_backend() in ('gpu','tpu'), 'Enable GPU in Runtime settings!'"
         ]},
        # Cell 2: Mount Google Drive (for persistent checkpoint storage)
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "# Mount Google Drive to save checkpoints persistently\n",
             "from google.colab import drive\n",
             "drive.mount('/content/drive')\n",
             "\n",
             "import os\n",
             "CKPT_DIR = '/content/drive/MyDrive/apollo_humanoid_checkpoints'\n",
             "os.makedirs(CKPT_DIR, exist_ok=True)\n",
             "print(f'Checkpoint dir: {CKPT_DIR}')"
         ]},
        # Cell 3: Download model
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": DOWNLOAD_CELL},
        # Cell 4: Training (CKPT_DIR injected via global)
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [TRAINING_CODE]}
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 0
    }

    nb_path = os.path.join(deploy_dir, "apollo_humanoid_colab_training.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[COLAB NOTEBOOK] {nb_path} ({os.path.getsize(nb_path):,} bytes)")
    print("-> Upload this file to Google Drive, then open with Colab")
    print(f"-> Or: https://colab.research.google.com/drive/  (upload & run)")

if __name__ == "__main__":
    generate()
