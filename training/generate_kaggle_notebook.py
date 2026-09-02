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
    # v9 — ROOT CAUSE ANALYSIS COMPLETE:
    #
    # DISCOVERED: Apollo keyframe "stand" has l_knee_fe=1.033 rad (~59 deg)
    # This means even "standing" pose has deeply bent knees by design!
    # The robot was NOT squatting — it WAS in correct pose but the reward
    # kernels were too tight (exp(-20*(z-1.016)^2) → nearly 0 even at
    # the correct keyframe pose due to body oscillation).
    #
    # REAL FIXES for v9:
    # FIX 1: Joint posture regularization — pull ALL joints to default_qpos
    #         (most important: prevents drift from keyframe pose)
    #
    # FIX 2: Multiplicative height GATING — height must be ≥ 95% nominal
    #         to receive ANY tracking reward (prevents true crouching)
    #
    # FIX 3: Softer height kernel exp(-5*(z-z_nom)^2) not exp(-20*...)
    #         sigma=0.32m vs 0.16m — wider tolerance since keyframe z=1.016
    #         but natural oscillation can drop to ~0.97m
    #
    # FIX 4: Anti-crouch: terminate if z < 0.94*z_nom (valid fall only)
    #
    # FIX 5: Add TORQUE penalty to make crouching expensive
    #
    # FIX 6: Higher entropy coef=0.02, lower LR=2e-4 with schedule
    # =============================================================

    TRAINING_CODE = r'''
import os, time, math
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 1 STANDING BALANCE (v9)")
print("  Fix: Posture regularization + Multiplicative gating")
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
        log_std = self.param("log_std",
                             nn.initializers.constant(-0.5),
                             (self.action_dim,))
        log_std = jnp.clip(log_std, -2.5, 0.5)
        value   = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value

# ================================================================
# 2. PHYSICS MODEL
# ================================================================
model_path = "mujoco_menagerie/apptronik_apollo/scene.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
mj_model.opt.timestep      = 0.002
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
default_qpos = jnp.array(mj_model.key_qpos[key_id])
default_ctrl = jnp.array(mj_model.key_ctrl[key_id])

Z_NOMINAL    = float(default_qpos[2])   # 1.016 m
ACTION_SCALE = 0.10
LPF_ALPHA    = 0.30
EPISODE_LEN  = 1000
# FIX 4: z < 94% → true fall (not crouching, since keyframe knee is 59 deg)
TERM_HEIGHT  = Z_NOMINAL * 0.94  # ~0.955 m
# Terminate if torso tilts past 20 degrees
TERM_TILT_COS = math.cos(20.0 * math.pi / 180.0)  # cos(20 deg) = 0.940

# Apollo knee joint qpos indices (discovered from model inspection)
# l_knee_fe = qpos[30], r_knee_fe = qpos[36]
# Default knee angle from keyframe = 1.033 rad (~59 deg) — by design
L_KNEE_IDX = 30 - 7   # offset from joint qpos start (after 7 free-joint dofs)
R_KNEE_IDX = 36 - 7
DEFAULT_KNEE = float(default_qpos[30])  # 1.033 rad

print(f"z_nominal = {Z_NOMINAL:.4f} m | term_height = {TERM_HEIGHT:.4f} m")
print(f"knee default = {DEFAULT_KNEE:.3f} rad ({math.degrees(DEFAULT_KNEE):.1f} deg)")

OBS_DIM = 1 + 4 + 3 + 3 + 3 + nu + nu + 3 * nu

def get_obs(d, action_hist):
    qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    pg = jnp.array([2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx),
                    -(qw**2 - qx**2 - qy**2 + qz**2)])
    jpos = d.qpos[7:7+nu] - default_qpos[7:7+nu]
    jvel = d.qvel[6:6+nu]
    obs  = jnp.concatenate([d.qpos[2:3], d.qpos[3:7], pg,
                             d.qvel[:3], d.qvel[3:6],
                             jpos, jvel, action_hist.ravel()])
    return jnp.clip(obs, -20.0, 20.0)

# ================================================================
# 3. REWARD — Multiplicative gating + joint posture regularization
# ================================================================
def compute_reward(qpos, qvel, ctrl, prev_lpf, action):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    up_z = -(qw**2 - qx**2 - qy**2 + qz**2)  # -1 when upright

    # FIX 3: softer height kernel (sigma=0.1m instead of 0.07m)
    r_height  = jnp.exp(-5.0 * jnp.square(qpos[2] - Z_NOMINAL))

    # FIX 2: height GATE — multiplicative, kills reward if z too low
    # Below 96% nominal → gate approaches 0 exponentially fast
    z_drop     = jnp.maximum(0.0, Z_NOMINAL * 0.96 - qpos[2])
    height_gate = jnp.exp(-200.0 * z_drop**2)

    # Upright and lateral tilt
    r_upright = jnp.exp(-8.0 * (1.0 - up_z))
    r_lateral = jnp.exp(-8.0 * (qpos[4]**2 + qpos[5]**2))

    # FIX 1: Joint posture regularization — pull to keyframe default
    # Most critical: legs must match keyframe knee=1.033rad, hip_fe=-0.477rad
    jpos  = qpos[7:7+nu]
    djpos = jpos - default_qpos[7:7+nu]
    # Weight leg joints 5x more than arm joints
    leg_mask = jnp.zeros(nu).at[20:].set(1.0)  # acts 20-31 = leg actuators
    arm_mask = 1.0 - leg_mask
    weighted_djpos = djpos * (5.0 * leg_mask + 1.0 * arm_mask)
    r_posture = jnp.exp(-2.0 * jnp.mean(jnp.square(weighted_djpos)))

    # Combine tracking: multiplicative gate on height, additive for others
    r_track = height_gate * (
        2.0 * r_height
        + 2.0 * r_upright
        + 1.0 * r_lateral
        + 2.0 * r_posture  # FIX 1: strong joint regularization
    )

    # Survival bonus: conditional on truly standing (z > 97% nominal)
    r_alive = 0.5 * (qpos[2] > Z_NOMINAL * 0.97).astype(jnp.float32)

    # Regularization penalties
    p_linvel   = jnp.sum(jnp.square(qvel[:2]))
    p_angvel   = jnp.sum(jnp.square(qvel[3:6]))
    p_jvel     = jnp.mean(jnp.square(qvel[6:6+nu]))
    p_act_rate = jnp.mean(jnp.square(action - prev_lpf))
    # FIX 5: torque penalty — makes crouching (high sustained torque) expensive
    norm_ctrl  = ctrl / jnp.maximum(1e-3, ctrl_range[:, 1])
    p_torque   = jnp.mean(jnp.square(norm_ctrl))
    # Joint limit penalty
    jpos_raw = qpos[7:7+nu]
    dist     = jnp.minimum(jpos_raw - ctrl_range[:nu,0],
                            ctrl_range[:nu,1] - jpos_raw)
    p_lim    = jnp.sum(jnp.square(jnp.minimum(dist, 0.0)))

    total = (
        r_track
        + r_alive
        - 1.0   * p_linvel
        - 0.2   * p_angvel
        - 0.005 * p_jvel
        - 0.01  * p_act_rate
        - 0.05  * p_torque   # FIX 5
        - 20.0  * p_lim
    )
    return total

# ================================================================
# 4. ENVIRONMENT
# ================================================================
def env_reset(rng):
    rng_q, rng_v = jax.random.split(rng)
    dq = jax.random.uniform(rng_q, (nq,), minval=-0.01, maxval=0.01)
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.02, maxval=0.02)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=default_qpos + dq, qvel=dv)
    d  = mjx.forward(mjx_model, d)
    ah = jnp.zeros((3, nu))
    lp = jnp.zeros(nu)
    return get_obs(d, ah), {"d": d, "ah": ah, "lpf": lp, "step": 0}

def env_step(state, raw_act):
    d, ah, prev_lpf = state["d"], state["ah"], state["lpf"]
    step = state["step"]

    lpf  = LPF_ALPHA * raw_act + (1.0 - LPF_ALPHA) * prev_lpf
    ctrl = jnp.clip(default_ctrl + lpf * ACTION_SCALE,
                    ctrl_range[:,0], ctrl_range[:,1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=5)

    rew  = compute_reward(d.qpos, d.qvel, ctrl, prev_lpf, raw_act)

    qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
    up_z       = -(qw**2 - qx**2 - qy**2 + qz**2)
    # FIX 4: strict z threshold, strict tilt threshold
    terminated = jnp.logical_or(
        d.qpos[2] < TERM_HEIGHT,
        up_z < TERM_TILT_COS)   # up_z < cos(20deg) → tilt > 20deg
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

    next_lpf = jnp.where(done, jnp.zeros(nu), lpf)
    next_ah  = jnp.concatenate([raw_act[None], ah[:2]], axis=0)
    obs      = get_obs(d, next_ah)
    nst      = {"d": d, "ah": next_ah, "lpf": next_lpf, "step": step_new}
    return obs, nst, rew, terminated, truncated

# ================================================================
# 5. PPO — FIX 6: ent_coef=0.02, LR schedule
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 32
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.02   # FIX 6: high entropy to prevent collapse
VF_COEF      = 0.5
MAX_GRAD     = 0.3
TOTAL_STEPS  = 100_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT  # 131,072

N_ITERS      = TOTAL_STEPS // STEPS_PER_IT
# FIX 6: LR schedule 3e-4 → 5e-5
lr_schedule  = optax.linear_schedule(3e-4, 5e-5, N_ITERS)

network   = ActorCritic(action_dim=nu)
rng       = jax.random.PRNGKey(42)
rng, ri   = jax.random.split(rng)
params    = network.init(ri, jnp.zeros((1, OBS_DIM)))
tx        = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD),
    optax.adam(lr_schedule, eps=1e-5))
opt_state = tx.init(params)

rng_envs = jax.random.split(rng, NUM_ENVS)
_, states = jax.vmap(env_reset)(rng_envs)

# Theoretical max reward for pass judgment
R_MAX = 2.0 + 2.0 + 1.0 + 2.0 + 0.5  # = 7.5 (r_track terms + r_alive)

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra = jax.random.split(r)
        obs  = jax.vmap(lambda s: get_obs(s["d"], s["ah"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        act  = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp   = -0.5 * jnp.sum(
            jnp.square((act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, act)
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(
        _step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    lobs = jax.vmap(lambda s: get_obs(s["d"], s["ah"]))(fst)
    _, _, nv = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        r, v, term, trunc = rews[t], vals[t], terms[t], truncs[t]
        done  = jnp.logical_or(term, trunc)
        delta = r + GAMMA * nxv * (1. - term.astype(jnp.float32)) - v
        gae   = delta + GAMMA * LAM * (1. - done.astype(jnp.float32)) * gae
        return (gae, v), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv),
                            jnp.arange(ROLLOUT - 1, -1, -1))
    advs  = jnp.flip(advs, axis=0)
    rets  = advs + vals
    advs  = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret = map(flat, [obs, act, old_lp, advs, rets])
    old_vals_flat = flat(vals)

    def loss_fn(p):
        mu, ls, v = network.apply(p, fo)
        std = jnp.exp(ls)
        lp  = -0.5 * jnp.sum(
            jnp.square((fa - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)
        log_ratio = jnp.clip(lp - flp, -5.0, 5.0)
        ratio = jnp.exp(log_ratio)
        pg   = -jnp.mean(jnp.minimum(
            ratio * fadv,
            jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * fadv))
        v_clip = old_vals_flat + jnp.clip(v - old_vals_flat, -5.0, 5.0)
        vf   = VF_COEF * jnp.mean(jnp.maximum(
            jnp.square(v - fret), jnp.square(v_clip - fret)))
        ent  = -ENT_COEF * jnp.mean(
            jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        total = pg + vf + ent
        return jnp.where(jnp.isnan(total), 0.0, total)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, upd)
    return params, opt_state, fst, rng, jnp.mean(rews), loss

# ================================================================
# 6. TRAINING LOOP
# ================================================================
os.makedirs("checkpoints", exist_ok=True)
t0, cur = time.time(), 0
PASS_THRESHOLD = R_MAX * 0.80
print(f"R_max={R_MAX:.1f} | PASS threshold (80%)={PASS_THRESHOLD:.2f}")

for it in range(1, N_ITERS + 1):
    t1 = time.time()
    params, opt_state, states, rng, mr, loss = \
        ppo_iter(params, opt_state, states, rng)
    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)
    pct = float(mr) / R_MAX * 100.0

    if it % 10 == 0 or it == 1:
        status = "*** PASS! ***" if float(mr) >= PASS_THRESHOLD else "..."
        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={float(mr):6.3f} ({pct:.1f}%/{PASS_THRESHOLD:.1f}) | "
              f"loss={float(loss):9.4f} | sps={sps:,.0f} | "
              f"t={time.time()-t0:.0f}s {status}", flush=True)

    if it % 50 == 0 or it == N_ITERS:
        ck = f"checkpoints/apollo_stage1_v9_step_{cur}.npz"
        jnp.savez(ck, **flax.traverse_util.flatten_dict(params, sep="/"))
        print(f"  -> checkpoint: {ck}", flush=True)

print("STAGE 1 v9 COMPLETE", flush=True)
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
    print(f"[v9 NOTEBOOK] {nb_path} ({os.path.getsize(nb_path):,} bytes)")

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
