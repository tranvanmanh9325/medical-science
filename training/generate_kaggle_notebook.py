import os
import json

def generate():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = os.path.join(root_dir, "kaggle_kernel_deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    with open(os.path.join(root_dir, "gpu", "kaggle.json")) as f:
        creds = json.load(f)
    username = creds.get("username", "manh090305")

    # =============================================================
    # v13 — FIX 2 ROOT-CAUSE BUGS confirmed by research:
    #
    # BUG 1 (ALL v11,v12): get_upvector Z sign WRONG
    #   Code:    -(qw²-qx²-qy²+qz²)
    #   When upright (qw=1,qx=qy=qz=0): returns -1.0 (WRONG!)
    #   Correct: +(qw²-qx²-qy²+qz²) = 1-2(qx²+qy²)
    #   Effect:  upvec[2]=-1 < TERM_TILT=0 → robot TERMINATED AT STEP 1
    #            ALWAYS → stays on floor → reward=-0.042 forever
    #   FIX:     return 1.0 - 2*(qx²+qy²) for Z component
    #
    # BUG 2 (ALL versions v11-v12): No auto-reset when done=True
    #   After termination, d (MJX state) is NOT reset to standing pose
    #   Robot lies on floor permanently → joints splayed out → cost_stand_still≈4.7
    #   FIX:     in scan loop, call env_reset with new rng and swap state
    #            using jax.tree.map + jnp.where when done=True
    #
    # ALSO FIX: TERM_TILT threshold from 0.0 → 0.5 (tilted >60°)
    #           TERM_HEIGHT from 0.2m → 0.7m (clear fall detection)
    # =============================================================

    TRAINING_CODE = r'''
import os, time, math
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 1 STANDING BALANCE (v13)")
print("  FIX: upvector sign + auto-reset on done")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "GPU required!"

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
model_path = "mujoco_menagerie/apptronik_apollo/scene.xml"
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
default_ctrl = jnp.array(mj_model.key_ctrl[key_id])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL      = float(default_qpos[2])
TRACKING_SIGMA = 0.25
ACTION_SCALE   = 0.3
EPISODE_LEN    = 1000
# FIX: tighter but reasonable fall thresholds
TERM_HEIGHT    = Z_NOMINAL * 0.75  # 75% nominal (same as v7 which worked)
TERM_TILT      = 0.5               # upvec_z < 0.5 → tilted > 60°

print(f"z_nominal={Z_NOMINAL:.4f}m | ctrl_dt={CTRL_DT:.3f}s | n_substeps={N_SUBSTEPS}")
print(f"term_height={TERM_HEIGHT:.3f}m | term_tilt_cos={TERM_TILT}")

# Verify upvector formula on identity quaternion
test_qpos = jnp.zeros(nq).at[3].set(1.0)  # identity: qw=1
OBS_DIM = 3 + 3 + 3 + nu + nu + nu

# BUG 1 FIX: correct upvector formula
# Body Z-axis in world = R@[0,0,1], where R is rotation matrix from quat
# For unit quat [qw,qx,qy,qz]: R[2,2] = qw²-qx²-qy²+qz² = 1-2(qx²+qy²)
def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([
        2.0 * (qx*qz + qw*qy),           # R[0,2]: X component of body-Z in world
        2.0 * (qy*qz - qw*qx),           # R[1,2]: Y component
        1.0 - 2.0 * (qx**2 + qy**2),    # R[2,2]: Z component (=+1 when upright)
    ])

# Sanity check
test_up = get_upvector(test_qpos)
print(f"[SANITY] upvector(identity quat) = {test_up} (should be [0,0,1])")

def get_obs(d, prev_act):
    upvec  = get_upvector(d.qpos)
    linvel = d.qvel[:3]
    angvel = d.qvel[3:6]
    jpos   = d.qpos[7:7+nu] - default_pose
    jvel   = d.qvel[6:6+nu]
    obs    = jnp.concatenate([upvec, linvel, angvel, jpos, jvel, prev_act])
    return jnp.clip(obs, -20.0, 20.0)

# ================================================================
# 3. REWARD — DeepMind formulation, NO clip to [0,inf)
# ================================================================
def compute_reward(qpos, qvel, torques, action, prev_action):
    upvec = get_upvector(qpos)

    tracking_lin_vel = jnp.exp(-jnp.sum(jnp.square(qvel[:2])) / TRACKING_SIGMA)
    tracking_ang_vel = jnp.exp(-jnp.square(qvel[5]) / TRACKING_SIGMA)
    cost_linvel_z    = jnp.square(qvel[2])
    cost_angvel_xy   = jnp.sum(jnp.square(qvel[3:5]))
    # BUG 1 FIX: now orientation cost is 0 when upright (upvec_xy≈0)
    cost_orientation = jnp.sum(jnp.square(upvec[:2]))
    jpos = qpos[7:7+nu]
    cost_stand_still = jnp.sum(jnp.abs(jpos - default_pose))
    cost_torques     = (jnp.sqrt(jnp.sum(jnp.square(torques))) +
                        jnp.sum(jnp.abs(torques)))
    cost_action_rate = jnp.sum(jnp.square(action - prev_action))
    out_lo = -jnp.clip(jpos - ctrl_range[:nu, 0], -jnp.inf, 0.0)
    out_hi =  jnp.clip(jpos - ctrl_range[:nu, 1],  0.0, jnp.inf)
    cost_dof_limits  = jnp.sum(out_lo + out_hi)

    total = (
        1.0   * tracking_lin_vel
        + 0.5  * tracking_ang_vel
        - 2.0  * cost_linvel_z
        - 0.05 * cost_angvel_xy
        - 1.0  * cost_orientation
        - 0.5  * cost_stand_still
        - 1e-4 * cost_torques
        - 0.01 * cost_action_rate
        - 10.0 * cost_dof_limits
    )
    return total * CTRL_DT  # scale but NO clip to [0,inf)

# ================================================================
# 4. ENVIRONMENT — with correct auto-reset
# ================================================================
def env_reset(rng):
    rng_q, rng_v, rng_j = jax.random.split(rng, 3)
    noise = jax.random.uniform(rng_j, (nq-7,), minval=0.85, maxval=1.15)
    qpos  = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.01, maxval=0.01),
        default_qpos[7:] * noise
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.05, maxval=0.05)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)
    prev_act = jnp.zeros(nu)
    return {"d": d, "prev_act": prev_act, "step": jnp.zeros((), jnp.int32)}

def env_step(state, action_and_rng):
    raw_act, rng_reset = action_and_rng
    d, prev_act, step = state["d"], state["prev_act"], state["step"]

    ctrl = jnp.clip(default_ctrl + raw_act * ACTION_SCALE,
                    ctrl_range[:,0], ctrl_range[:,1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)

    torques = d.qfrc_actuator[6:6+nu]
    rew     = compute_reward(d.qpos, d.qvel, torques, raw_act, prev_act)
    obs_out = get_obs(d, raw_act)  # obs after step (before reset)

    upvec      = get_upvector(d.qpos)
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

    # BUG 2 FIX: auto-reset state when done
    reset_state = env_reset(rng_reset)
    next_d    = jax.tree.map(
        lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act  = jnp.where(done, jnp.zeros(nu), raw_act)
    next_step = jnp.where(done, jnp.zeros((), jnp.int32), step_new)

    nst = {"d": next_d, "prev_act": next_act, "step": next_step}
    return obs_out, nst, rew, terminated, truncated

# ================================================================
# 5. PPO — pass rng keys for auto-reset
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

# Initialize states
rng_envs = jax.random.split(rng, NUM_ENVS)
states   = jax.vmap(env_reset)(rng_envs)

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        # Split reset rngs per environment for auto-reset
        r_resets = jax.random.split(r_reset, NUM_ENVS)

        obs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        act  = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp   = -0.5 * jnp.sum(
            jnp.square((act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)

        # env_step now takes (action, rng_reset) as inputs
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, r_resets))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(
        _step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(fst)
    _, _, nv_last = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        r, v, term, trunc = rews[t], vals[t], terms[t], truncs[t]
        done  = jnp.logical_or(term, trunc)
        delta = r + GAMMA * nxv * (1. - term.astype(jnp.float32)) - v
        gae   = delta + GAMMA * LAM * (1. - done.astype(jnp.float32)) * gae
        return (gae, v), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last),
                            jnp.arange(ROLLOUT - 1, -1, -1))
    advs  = jnp.flip(advs, axis=0)
    rets  = advs + vals
    advs  = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret = map(flat, [obs, act, old_lp, advs, rets])
    ovf = flat(vals)

    def loss_fn(p):
        mu, ls, v = network.apply(p, fo)
        std = jnp.exp(ls)
        lp  = -0.5 * jnp.sum(
            jnp.square((fa - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)
        lr_  = jnp.clip(lp - flp, -5.0, 5.0)
        ratio = jnp.exp(lr_)
        pg   = -jnp.mean(jnp.minimum(ratio * fadv,
                                      jnp.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)*fadv))
        vc   = ovf + jnp.clip(v - ovf, -5.0, 5.0)
        vf   = VF_COEF * jnp.mean(jnp.maximum(
            jnp.square(v-fret), jnp.square(vc-fret)))
        ent  = -ENT_COEF * jnp.mean(
            jnp.sum(ls + 0.5*math.log(2*math.pi*math.e), axis=-1))
        total = pg + vf + ent
        return jnp.where(jnp.isnan(total), 0.0, total)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, upd)
    return params, opt_state, fst, rng, jnp.mean(rews), loss

# ================================================================
# 6. TRAINING LOOP
#    Expected: reward increases from ~-0.03 toward +0.005 to +0.010
#    when robot learns to stand (tracking≈+1.5, orientation≈0, stand_still small)
# ================================================================
os.makedirs("checkpoints", exist_ok=True)
t0, cur = time.time(), 0

for it in range(1, N_ITERS + 1):
    t1 = time.time()
    params, opt_state, states, rng, mr, loss = \
        ppo_iter(params, opt_state, states, rng)
    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)

    if it % 10 == 0 or it == 1:
        r_val = float(mr)
        if r_val > 0.005:   status = "*** STANDING ***"
        elif r_val > -0.01: status = "near-standing"
        elif r_val > -0.03: status = "improving"
        else:                status = "..."
        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={r_val:.5f} | loss={float(loss):.4f} | "
              f"sps={sps:,.0f} | t={time.time()-t0:.0f}s {status}", flush=True)

    if it % 50 == 0 or it == N_ITERS:
        ck = f"checkpoints/apollo_stage1_v13_step_{cur}.npz"
        jnp.savez(ck, **flax.traverse_util.flatten_dict(params, sep="/"))
        print(f"  -> checkpoint: {ck}", flush=True)

print("STAGE 1 v13 COMPLETE", flush=True)
'''

    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["!nvidia-smi\n", "import os; print('CWD:', os.getcwd())"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "!pip install -q --no-cache-dir mujoco mujoco-mjx flax optax\n",
             "import jax\n",
             "print('Backend:', jax.default_backend(), '| Devices:', jax.devices())\n",
             "assert jax.default_backend() in ('gpu','tpu'), 'GPU required!'"
         ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "import os\n",
             "if not os.path.exists('mujoco_menagerie'):\n",
             "    os.system('git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git')\n",
             "assert os.path.exists('mujoco_menagerie/apptronik_apollo/scene.xml')\n",
             "print('[OK] Apollo model ready.')"
         ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [TRAINING_CODE]}
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }

    nb_path = os.path.join(deploy_dir, "apollo_humanoid_mjx_training.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[v13 NOTEBOOK] {nb_path} ({os.path.getsize(nb_path):,} bytes)")

    meta = {
        "id": f"{username}/apollo-humanoid-mjx-training-dual-t4",
        "title": "apollo-humanoid-mjx-training-dual-t4",
        "code_file": "apollo_humanoid_mjx_training.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": []
    }
    with open(os.path.join(deploy_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print("[METADATA UPDATED]")

if __name__ == "__main__":
    generate()
