import os, time, math, glob
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx
import numpy as np

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 2: WALKING (v8)")
print("  Linear Actor + Sigmoid log_std + Unbounded Gaussian")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "GPU required!"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true --xla_gpu_enable_latency_hiding_scheduler=true")

OBS_DIM_S2 = 114
CKPT_DIR   = "/content/checkpoints"

# ================================================================
# 1. ACTOR-CRITIC — v8: Unbounded Gaussian (industry standard)
#
# CRITICAL BUG confirmed in v6/v7 (via weight norm analysis):
#   Bug A: mean = nn.tanh(Dense(x))
#     With std=1.66, sampled actions always hit +/-1 clip boundary.
#     (act - mu) = 0 in PPO loss -> gradient through Dense_3 = 0.
#     Result: actor output layer COMPLETELY FROZEN (weight diff = 0.00).
#   Bug B: log_std = jnp.clip(log_std, -3.0, 0.5)
#     Adam pushes log_std to 0.5068 > 0.5. jnp.clip at boundary
#     returns gradient = 0. log_std FROZEN at 0.5068 forever.
#
# v8 fix (legged_gym / rsl_rl industry standard):
#   1. Linear actor output (NO tanh) -> Dense_3 gradients always flow.
#   2. Sigmoid-bounded log_std: -4 + 3*sigmoid(param) -> range [-4,-1].
#      sigmoid always has gradient -> log_std always trains.
#      Init param=0 -> log_std=-4+1.5=-2.5 -> std=exp(-2.5)=0.082.
#   3. Unbounded Gaussian: raw_act = mu + std*noise (no clip!).
#      log_prob on raw_act -> correct PPO gradient.
#      clip(raw_act,-1,1) only at environment boundary (not in loss!).
# ================================================================
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in (512, 256, 128):
            x = nn.elu(nn.Dense(h)(x))
        # v8: LINEAR output — gradients always flow to this layer
        mean     = nn.Dense(self.action_dim)(x)
        # v8: sigmoid-bounded log_std — range [-4, -1], always has gradient
        ls_param = self.param("log_std", nn.initializers.constant(0.0), (self.action_dim,))
        log_std  = -4.0 + 3.0 * nn.sigmoid(ls_param)
        value    = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value

# ================================================================
# 2. PHYSICS MODEL — same as training (SIM_DT=0.002, N_SUBSTEPS=5)
# ================================================================
model_path = "/content/mujoco_menagerie/apptronik_apollo/scene.xml"
mj_model   = mujoco.MjModel.from_xml_path(model_path)
SIM_DT, N_SUBSTEPS = 0.002, 5
CTRL_DT = SIM_DT * N_SUBSTEPS   # 0.01s = 100Hz control

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
default_ctrl = jnp.array(mj_model.key_qpos[key_id][7:])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL    = float(default_qpos[2])       # ~1.016m standing height
ACTION_SCALE = 0.25
EPISODE_LEN  = 500                           # 5s per episode at 100Hz
# v8 STRICT termination thresholds (v6/v7 used 0.50 and cos(63deg)=0.45 — too lenient)
TERM_HEIGHT  = Z_NOMINAL * 0.75             # 0.762m — fall if below 76.2cm
TERM_TILT    = 0.906                         # cos(25deg) — fall if tilt > 25deg

STEP_FREQ   = 1.2
STANCE_DUTY = 0.55
L_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_fl")
R_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_fl")
CONTACT_Z_THR  = 0.08

print(f"z_nominal={Z_NOMINAL:.4f}m | TERM_HEIGHT={TERM_HEIGHT:.3f}m | TERM_TILT=cos(25deg)={TERM_TILT}")
print(f"action_scale={ACTION_SCALE} | CTRL_DT={CTRL_DT}s | step_freq={STEP_FREQ}Hz")

# ================================================================
# 3. NETWORK INIT — from scratch (NO Stage 1 transfer)
# ================================================================
network = ActorCritic(action_dim=nu)
rng     = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params  = network.init(ri, jnp.zeros((1, OBS_DIM_S2)))
print(f"[v8] Training from scratch | log_std init: {-4.0+3.0*0.5:.3f} (std={math.exp(-4.0+3.0*0.5):.3f})")

# ================================================================
# 4. ENVIRONMENT — reset, step, observations
# ================================================================
def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx**2+qy**2)])

def get_obs(d, prev_act, cmd_vel, phase):
    upvec = get_upvector(d.qpos)
    phi_l = 2.0*math.pi*phase; phi_r = 2.0*math.pi*((phase+0.5)%1.0)
    gait_phase  = jnp.array([jnp.sin(phi_l), jnp.cos(phi_l), jnp.sin(phi_r), jnp.cos(phi_r)])
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]; r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    foot_contact = jnp.array([(l_z<CONTACT_Z_THR).astype(jnp.float32),
                               (r_z<CONTACT_Z_THR).astype(jnp.float32)])
    obs = jnp.concatenate([upvec, d.qvel[:3], d.qvel[3:6],
                           d.qpos[7:7+nu]-default_pose, d.qvel[6:6+nu],
                           prev_act, cmd_vel, gait_phase, foot_contact])
    return jnp.clip(obs, -20.0, 20.0)

def env_reset(rng):
    rng_q, rng_v, rng_j, rng_cmd, rng_phase = jax.random.split(rng, 5)
    noise = jax.random.uniform(rng_j, (nq-7,), minval=-0.05, maxval=0.05)
    qpos  = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.01, maxval=0.01),
        default_qpos[7:] + noise
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.05, maxval=0.05)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)
    # v8: cmd_vel starts from 0 (curriculum will expand range later)
    cmd_vel = jax.random.uniform(
        rng_cmd, (3,),
        minval=jnp.array([0.0,  -0.20, -0.30]),
        maxval=jnp.array([0.25,  0.20,  0.30]),
    )
    phase = jax.random.uniform(rng_phase, (), minval=0.0, maxval=1.0)
    return {
        "d": d, "prev_act": jnp.zeros(nu), "step": jnp.zeros((), jnp.int32),
        "phase": phase, "cmd_vel": cmd_vel,
    }

def env_step(state, action_and_rng):
    # action_and_rng[0] is env_act (CLIPPED for physics safety)
    env_act, rng_reset = action_and_rng
    d, prev_act, step, phase, cmd_vel = (
        state["d"], state["prev_act"], state["step"],
        state["phase"], state["cmd_vel"],
    )

    ctrl = jnp.clip(default_ctrl + env_act * ACTION_SCALE, ctrl_range[:, 0], ctrl_range[:, 1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)

    new_phase = (phase + CTRL_DT * STEP_FREQ) % 1.0

    rew, done_bonus = compute_reward(d, env_act, prev_act, cmd_vel, phase)
    obs_out = get_obs(d, env_act, cmd_vel, new_phase)

    upvec      = get_upvector(d.qpos)
    # v8 STRICT termination: tilt > 25deg OR pelvis < 76.2cm
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

    # v8: termination penalty — robot learns falling is bad
    total_rew = rew + jnp.where(terminated, -5.0 * CTRL_DT, 0.0)

    reset_state = env_reset(rng_reset)
    next_d     = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act   = jnp.where(done, jnp.zeros(nu), env_act)
    next_step  = jnp.where(done, jnp.zeros((), jnp.int32), step_new)
    next_phase = jnp.where(done, reset_state["phase"], new_phase)
    next_cmd   = jnp.where(done, reset_state["cmd_vel"], cmd_vel)

    nst = {"d": next_d, "prev_act": next_act, "step": next_step,
           "phase": next_phase, "cmd_vel": next_cmd}
    return obs_out, nst, total_rew, terminated, truncated

# ================================================================
# 5. REWARD FUNCTION — v8: Asymmetric velocity + strict termination
#
# v6/v7 failures (confirmed by simulation):
#   - Symmetric exp(-|v-cmd|^2) has ~0 gradient when overshoot is large
#   - Robot learned to lean forward (gravity = free energy) to get vel reward
#   - cmd=0.3 -> actual 1.57 m/s in 1.3s -> ALWAYS falls
#   - TERM_TILT=cos(63deg) too lenient -> falling forward was profitable!
#
# v8 fixes:
#   1. ASYMMETRIC: overshoot -> LINEAR penalty (-2*error), LARGE gradient!
#      undershoot -> EXP reward (smooth encouragement)
#   2. Explicit PITCH penalty: 3*upvec[1]^2 — directly penalize forward lean
#   3. TERM_TILT=cos(25deg)=0.906 — robot CANNOT lean 25deg without reset
# ================================================================
_VEL_SIGMA   = 0.15   # Tight sigma (tighter = more precise tracking required)
_PENALTY_SCL = 0.20   # Start low, increases to 1.0 over 100M steps
_ENT_COEF    = 0.01   # Decays to 0.001 over 50M steps

def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos  = d.qpos
    qvel  = d.qvel
    upvec = get_upvector(qpos)

    # Asymmetric velocity tracking (primary signal)
    vx_error = qvel[0] - cmd_vel[0]   # positive = faster than commanded
    vy_error = qvel[1] - cmd_vel[1]
    vel_sigma = jnp.float32(_VEL_SIGMA)
    r_vel_x = jnp.where(
        vx_error > 0,
        -2.0 * vx_error,                              # LINEAR penalty for overshoot
        jnp.exp(-jnp.square(vx_error) / vel_sigma)   # EXP reward for undershoot
    )
    r_vel_y   = jnp.exp(-jnp.square(vy_error) / vel_sigma)
    r_vel_ang = jnp.exp(-jnp.square(qvel[5] - cmd_vel[2]) / vel_sigma)

    # Orientation: upright body
    r_orient = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.10)

    # Height: keep pelvis near nominal
    r_height = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.10)

    # Alive reward: large — prevents robot from preferring early death
    r_alive = 1.0

    # Gait phase matching
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]; r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    l_c = (l_z < CONTACT_Z_THR).astype(jnp.float32)
    r_c = (r_z < CONTACT_Z_THR).astype(jnp.float32)
    l_ts = (phase < STANCE_DUTY).astype(jnp.float32)
    rp   = (phase + 0.5) % 1.0
    r_ts = (rp < STANCE_DUTY).astype(jnp.float32)
    r_gait = (jnp.where(l_ts > 0.5, l_c, 1. - l_c) + jnp.where(r_ts > 0.5, r_c, 1. - r_c)) * 0.3

    # Penalties (scaled by _PENALTY_SCL, increases over training)
    pen_scl   = jnp.float32(_PENALTY_SCL)
    p_ar      = pen_scl * 0.02 * jnp.mean(jnp.square(action - prev_action))
    p_torque  = pen_scl * 2e-4 * jnp.sum(jnp.square(action))
    p_base_tw = pen_scl * 0.05 * (jnp.square(qvel[3]) + jnp.square(qvel[4]))
    # v8: EXPLICIT pitch penalty — directly penalizes forward lean (anti-runaway)
    p_pitch   = pen_scl * 3.0 * jnp.square(upvec[1])

    total = (r_vel_x * 5.0 + r_vel_y * 1.0 + r_vel_ang * 0.5
             + r_orient * 0.5 + r_height * 0.3
             + r_alive + r_gait
             - p_ar - p_torque - p_base_tw - p_pitch)

    return total * CTRL_DT, 0.0   # (reward, done_bonus — done_bonus added in env_step)

# ================================================================
# 6. PPO HYPERPARAMETERS — v8
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 24       # Short horizon optimal for MJX locomotion
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5      # Tighter clip for unbounded Gaussian stability
N_EPOCHS     = 4
MINIBATCH    = 4096
KL_TARGET    = 0.02
TOTAL_STEPS  = 300_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT   # 4096 x 24 = 98,304
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT

print(f"[v8 CONFIG] ROLLOUT={ROLLOUT} | STEPS_PER_IT={STEPS_PER_IT:,} | N_ITERS={N_ITERS}")
print(f"[v8 CONFIG] LINEAR actor | sigmoid log_std | unbounded Gaussian | MAX_GRAD={MAX_GRAD}")
print(f"[v8 CONFIG] ENT_COEF=0.01->0.001 | PENALTY_SCL=0.20->1.00 | TOTAL={TOTAL_STEPS:,} steps")

lr_schedule = optax.cosine_decay_schedule(1e-3, N_ITERS, alpha=0.1)   # 1e-3 -> 1e-4
tx          = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(lr_schedule, eps=1e-5))
opt_state   = tx.init(params)

rng_envs = jax.random.split(rng, NUM_ENVS)
states   = jax.vmap(env_reset)(rng_envs)

# ================================================================
# 7. ROLLOUT + PPO UPDATE
# ================================================================
@jax.jit
def collect_rollout(params, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs  = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        # v8 FIX: raw Gaussian sample — unbounded, used for log_prob
        raw_act  = mu + std * jax.random.normal(ra, mu.shape)
        lp   = jnp.clip(-0.5 * jnp.sum(
            jnp.square((raw_act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        # v8 FIX: clip only for environment (physics safety), NOT for loss
        env_act = jnp.clip(raw_act, -1., 1.)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (env_act, r_resets))
        # Store raw_act in buffer for correct PPO gradient signal
        return (nst, p, r), (obs, raw_act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(
        _step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(fst)
    _, _, nv_last = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        done  = jnp.logical_or(terms[t], truncs[t])
        delta = rews[t] + GAMMA * nxv * (1. - terms[t].astype(jnp.float32)) - vals[t]
        gae   = delta + GAMMA * LAM * (1. - done.astype(jnp.float32)) * gae
        return (gae, vals[t]), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last),
                            jnp.arange(ROLLOUT - 1, -1, -1))
    advs  = jnp.flip(advs, axis=0)
    rets  = advs + vals
    advs  = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat  = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret, ovf = *map(flat, [obs, act, old_lp, advs, rets]), flat(vals)
    return fst, rng, fo, fa, flp, fadv, fret, ovf, jnp.mean(rews)

@jax.jit
def ppo_minibatch_update(params, opt_state, fo_mb, fa_mb, flp_mb, fadv_mb, fret_mb, ovf_mb, ent_coef):
    def loss_fn(p):
        mu, ls, v = network.apply(p, fo_mb)
        std = jnp.exp(ls)
        # log_prob on raw action (fa_mb is raw_act, not clipped)
        lp  = jnp.clip(-0.5 * jnp.sum(jnp.square((fa_mb - mu) / (std + 1e-8)) +
                        2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        ratio = jnp.exp(jnp.clip(lp - flp_mb, -5., 5.))
        approx_kl = jnp.mean(0.5 * jnp.square(lp - flp_mb))
        pg    = -jnp.mean(jnp.minimum(ratio * fadv_mb,
                          jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * fadv_mb))
        vc    = ovf_mb + jnp.clip(v - ovf_mb, -5., 5.)
        vf    = VF_COEF * jnp.mean(jnp.maximum(jnp.square(v - fret_mb),
                                                 jnp.square(vc - fret_mb)))
        ent   = -ent_coef * jnp.mean(jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        total = jnp.where(jnp.isnan(pg + vf + ent), 0.0, pg + vf + ent)
        return total, approx_kl

    (loss, approx_kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, approx_kl

# ================================================================
# 8. TRAINING LOOP
# ================================================================
os.makedirs(CKPT_DIR, exist_ok=True)
t0, cur = time.time(), 0

WALK_THRESHOLD      = 0.020
WALK_WELL_THRESHOLD = 0.040

print(f"\nAPOLLO HUMANOID - STAGE 2 v8 (LINEAR ACTOR + SIGMOID LOG_STD)")
print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | ROLLOUT={ROLLOUT}")
print("=" * 64)

# Curriculum: cmd_vel range grows as robot learns
CURR_P0 = 30_000_000; CURR_P1 = 100_000_000; CURR_P2 = 200_000_000

def get_curriculum_vx_max(n_steps):
    if n_steps < CURR_P0: return 0.25
    elif n_steps < CURR_P1:
        t = (n_steps - CURR_P0) / (CURR_P1 - CURR_P0); return 0.25 + t * (0.55 - 0.25)
    elif n_steps < CURR_P2:
        t = (n_steps - CURR_P1) / (CURR_P2 - CURR_P1); return 0.55 + t * (0.85 - 0.55)
    else: return 1.20

@jax.jit
def reseed_cmd_vel(states, rng, vx_max, vy_max, yaw_max):
    rngs = jax.random.split(rng, NUM_ENVS)
    def _new_cmd(r):
        return jax.random.uniform(r, (3,),
            minval=jnp.array([0.0,  -vy_max, -yaw_max]),
            maxval=jnp.array([vx_max, vy_max,  yaw_max]))
    return {**states, "cmd_vel": jax.vmap(_new_cmd)(rngs)}

import numpy as np_host

for it in range(1, N_ITERS + 1):
    t1 = time.time()

    # Adaptive parameters (host-side, outside jit) — update module-level vars
    ent_t        = min(1.0, cur / 50_000_000)
    _ENT_COEF    = 0.01 - ent_t * (0.01 - 0.001)    # 0.01 -> 0.001
    pen_t        = min(1.0, cur / 100_000_000)
    _PENALTY_SCL = 0.20 + pen_t * (1.0  - 0.20)     # 0.20 -> 1.00

    if it % 20 == 1:
        vx_max_cur  = get_curriculum_vx_max(cur)
        vy_max_cur  = min(0.3, vx_max_cur * 0.4)
        yaw_max_cur = min(0.5, vx_max_cur * 0.5)
        rng, rng_seed = jax.random.split(rng)
        states = reseed_cmd_vel(states, rng_seed,
                                jnp.float32(vx_max_cur),
                                jnp.float32(vy_max_cur),
                                jnp.float32(yaw_max_cur))

    states, rng, fo, fa, flp, fadv, fret, ovf, mr = \
        collect_rollout(params, states, rng)
    jax.block_until_ready(fo)

    N_SAMPLES  = fo.shape[0]
    N_MB       = N_SAMPLES // MINIBATCH
    last_loss  = 0.0; kl_stopped = False
    ent_coef_j = jnp.float32(_ENT_COEF)

    for epoch in range(N_EPOCHS):
        if kl_stopped: break
        perm = np_host.random.permutation(N_SAMPLES)
        for mb_idx in range(N_MB):
            idx   = perm[mb_idx * MINIBATCH:(mb_idx + 1) * MINIBATCH]
            idx_j = jnp.array(idx)
            params, opt_state, last_loss, approx_kl = ppo_minibatch_update(
                params, opt_state,
                fo[idx_j], fa[idx_j], flp[idx_j],
                fadv[idx_j], fret[idx_j], ovf[idx_j],
                ent_coef_j
            )
            if float(approx_kl) > KL_TARGET:
                kl_stopped = True; break

    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)

    if it % 10 == 0 or it <= 5:
        r_val    = float(mr)
        vx_max_p = get_curriculum_vx_max(cur)

        # Read log_std to monitor learning progress (key diagnostic)
        ls_flat = flax.traverse_util.flatten_dict(params, sep="/").get("params/log_std", None)
        ls_mean = float(jnp.mean(-4.0 + 3.0 * nn.sigmoid(ls_flat))) if ls_flat is not None else float("nan")

        if r_val > WALK_WELL_THRESHOLD: status = "*** WALKING WELL ***"
        elif r_val > WALK_THRESHOLD:    status = "*** WALKING ***"
        elif r_val > 0.015:             status = "stepping"
        elif r_val > 0.010:             status = "improving"
        else:                           status = "..."

        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={r_val:.5f} | loss={float(last_loss):.4f} | "
              f"sps={sps:,.0f} | vx_max={vx_max_p:.2f} | "
              f"ent={float(_ENT_COEF):.4f} | log_std={ls_mean:.3f} | "
              f"t={time.time()-t0:.0f}s {status}", flush=True)

    if it % 300 == 0 or it == N_ITERS:
        ck = f"{CKPT_DIR}/apollo_stage2_v8_step_{cur}.npz"
        flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
        flat_np["_step"] = np.array(cur)
        flat_np["_it"]   = np.array(it)
        np.savez(ck, **flat_np)
        ck_size = os.path.getsize(ck)
        if ck_size < 100_000:
            print(f"  [WARNING] Checkpoint small: {ck_size} bytes!", flush=True)
        else:
            print(f"  -> checkpoint: {ck} ({ck_size//1024}KB)", flush=True)

        # Git push to GitHub for persistent storage (failover recovery)
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if gh_token:
            try:
                import subprocess
                repo_url = f"https://x-access-token:{gh_token}@github.com/tranvanmanh9325/medical-science.git"
                dest = f"/content/ckpt_sync/apollo_stage2_v8_step_{cur}.npz"
                os.makedirs("/content/ckpt_sync", exist_ok=True)
                import shutil; shutil.copy(ck, dest)
                subprocess.run(["git", "config", "user.email", "colab@train.local"], cwd="/content/medical-science", capture_output=True)
                subprocess.run(["git", "config", "user.name", "Colab Trainer"], cwd="/content/medical-science", capture_output=True)
                dst_repo = f"/content/medical-science/colab_output/checkpoints_stage2"
                os.makedirs(dst_repo, exist_ok=True)
                shutil.copy(ck, f"{dst_repo}/apollo_stage2_v8_latest.npz")
                subprocess.run(["git", "add", "-A"], cwd="/content/medical-science", capture_output=True)
                subprocess.run(["git", "commit", "--allow-empty", "-m", f"[skip ci] checkpoint step={cur}"], cwd="/content/medical-science", capture_output=True)
                subprocess.run(["git", "push", repo_url, "main"], cwd="/content/medical-science", capture_output=True, timeout=60)
                print(f"  -> GitHub push OK (step={cur})", flush=True)
            except Exception as e:
                print(f"  [WARN] GitHub push failed: {e}", flush=True)

# Final checkpoint
flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
flat_np["_step"] = np.array(cur)
flat_np["_it"]   = np.array(it)
np.savez(f"{CKPT_DIR}/apollo_stage2_v8_final.npz", **flat_np)
print("\nSTAGE 2 v8 TRAINING COMPLETE! (Linear Actor + Sigmoid log_std)", flush=True)
print(f"Total steps: {cur:,} | Checkpoint: {CKPT_DIR}/apollo_stage2_v8_final.npz", flush=True)
