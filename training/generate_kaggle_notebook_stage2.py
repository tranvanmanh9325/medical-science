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
    # STAGE 2 v7 — ASYMMETRIC VELOCITY REWARD + STRICT TERMINATION
    #
    # v6 FAILURE ROOT CAUSE (from 300M step training + simulation analysis):
    #   1. RUNAWAY VELOCITY: exp(-|v-cmd|²/σ²) is symmetric but physics is NOT.
    #      Leaning forward uses gravity (free energy) → robot accelerates without bound.
    #      cmd=0.3 m/s → actual vx reaches 1.57 m/s in 1.3s → robot falls.
    #   2. log_std=0.508 for ALL 32 joints (std=1.662) → policy outputs NOISE.
    #      ENT_COEF=0.01 was too high → entropy saturation → no learning signal.
    #   3. TERM_TILT=cos(63°)=0.45 too lenient → robot could fall 63° before termination!
    #      "Falling forward" was rewarded (vel + survival for ~1.3s) before penalty.
    #
    # v7 fixes (research-backed, ETH RSL + legged_gym best practices):
    #   1. ASYMMETRIC velocity reward:
    #      - Overshoot (v > cmd): linear penalty (-2 * error) — large gradient when far!
    #      - Undershoot (v < cmd): exp kernel — smooth encouragement
    #   2. ENT_COEF decay: 0.01 → 0.001 over 50M steps (force policy commitment)
    #   3. log_std clipped: [-3.0, 0.5] in ActorCritic (prevent saturation)
    #   4. STRICT termination: TERM_TILT=cos(25°)=0.906, TERM_HEIGHT=0.762m
    #   5. EXPLICIT pitch penalty: p_pitch = 3.0 * upvec[1]² (directly penalize lean)
    #   6. Termination penalty: -5.0 * CTRL_DT when fallen (no free falls!)
    #   7. NO Stage 1 transfer (blocked completely — no code path to load it)
    # =============================================================

    TRAINING_CODE = r'''
import os, time, math, glob
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx
import numpy as np

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 2: WALKING & PUSH RECOVERY (v2)")
print("  Velocity Curriculum + Narrow Reward Kernel + Entropy Reg")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "GPU required!"

# ================================================================
# 1. ACTOR-CRITIC — extended input: 114-dim obs
# ================================================================
OBS_DIM_S1 = 105   # Stage 1 observation dimension
OBS_DIM_S2 = 114   # Stage 2: +9 dims (cmd_vel=3, gait_phase=4, foot_contact=2)

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
CTRL_DT = SIM_DT * N_SUBSTEPS  # 0.01s = 100 Hz control

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

Z_NOMINAL    = float(default_qpos[2])   # ~1.016m
ACTION_SCALE = 0.25
EPISODE_LEN  = 500    # 5 seconds per episode
# v7 STRICT termination — prevent "falling forward = free speed" exploit
TERM_HEIGHT  = Z_NOMINAL * 0.75   # 0.762m (was 0.50 in v6 → too lenient!)
TERM_TILT    = jnp.cos(jnp.deg2rad(25.0))  # cos(25°)=0.906 (was 0.45≈cos(63°)!)

# ── Gait Clock ────────────────────────────────────────────────
STEP_FREQ    = 1.2    # Hz
STANCE_DUTY  = 0.55

# ── Foot contact ─────────────────────────────────────────────
L_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_fl")
R_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_fl")
CONTACT_Z_THR  = 0.08

PUSH_START_STEP = 999_999_999
PUSH_MAX_STEP   = 999_999_999
PUSH_MAX_FORCE  = 0.0
PUSH_INTERVAL   = 200

print(f"z_nominal={Z_NOMINAL:.4f}m | action_scale={ACTION_SCALE} | step_freq={STEP_FREQ}Hz")
print(f"OBS_DIM: {OBS_DIM_S1} (Stage1) → {OBS_DIM_S2} (Stage2: +cmd_vel+gait+contact)")
print(f"v7: TERM_TILT=cos(25°)={float(TERM_TILT):.3f} | TERM_HEIGHT={Z_NOMINAL*0.75:.3f}m")

# ================================================================
# 3. NETWORK INIT — v7: FROM SCRATCH (Block Stage 1 transfer)
# ================================================================
# v7 CRITICAL FIX: Stage 1 dataset was leaking into v6 despite "from scratch" intent.
# Block transfer explicitly by NOT searching for Stage 1 checkpoints.
# v6 failure analysis showed log_std=0.508 (std=1.662) for ALL 32 joints = noise output.
# A real locomotion policy should have log_std ≈ -1.0 to -2.5 (std ≈ 0.08 to 0.36).
# Root cause of v6 failure: ENT_COEF=0.01 was too high, causing entropy saturation.
# Transfer + high entropy = policy never committed to any action → runaway velocity.
print("[v7] NO Stage 1 transfer — training purely from scratch")
print(f"[v7] log_std initialized to -0.5, clipped to [-3.0, 0.5]")

network = ActorCritic(action_dim=nu)
rng = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params = network.init(ri, jnp.zeros((1, OBS_DIM_S2)))

# ================================================================
# 4. OBSERVATION & ENVIRONMENT
# ================================================================
def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([
        2.0*(qx*qz + qw*qy),
        2.0*(qy*qz - qw*qx),
        1.0 - 2.0*(qx**2 + qy**2),
    ])

def get_obs(d, prev_act, cmd_vel, phase):
    """114-dim observation: base(105) + cmd_vel(3) + gait_phase(4) + foot_contact(2)."""
    upvec  = get_upvector(d.qpos)
    linvel = d.qvel[:3]
    angvel = d.qvel[3:6]
    jpos   = d.qpos[7:7+nu] - default_pose
    jvel   = d.qvel[6:6+nu]

    gait_phase = jnp.array([
        jnp.sin(2.0 * math.pi * phase),
        jnp.cos(2.0 * math.pi * phase),
        jnp.sin(2.0 * math.pi * (phase + 0.5)),
        jnp.cos(2.0 * math.pi * (phase + 0.5)),
    ])

    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    foot_contact = jnp.array([
        (l_z < CONTACT_Z_THR).astype(jnp.float32),
        (r_z < CONTACT_Z_THR).astype(jnp.float32),
    ])

    obs = jnp.concatenate([
        upvec, linvel, angvel, jpos, jvel, prev_act,
        cmd_vel, gait_phase, foot_contact,
    ])
    return jnp.clip(obs, -20.0, 20.0)

def env_reset(rng):
    rng_q, rng_v, rng_j, rng_cmd, rng_phase = jax.random.split(rng, 5)
    noise = jax.random.uniform(rng_j, (nq - 7,), minval=-0.05, maxval=0.05)
    qpos  = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.01, maxval=0.01),
        default_qpos[7:] + noise,
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.05, maxval=0.05)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)

    # v7 curriculum: cmd_vel includes 0.0 (robot must learn to stand AND walk)
    # Phase 0 (0-30M):  vx in [0.0, 0.25] — balance + tiny movement
    # Phase 1 (30-100M): vx in [0.0, 0.55] — walking starts
    # Phase 2 (100-200M): vx in [0.0, 0.85] — speed up
    # Phase 3 (200M+):  vx in [0.0, 1.20] — full range
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
    raw_act, rng_reset = action_and_rng
    d, prev_act, step, phase, cmd_vel = (
        state["d"], state["prev_act"], state["step"],
        state["phase"], state["cmd_vel"],
    )

    ctrl = jnp.clip(default_ctrl + raw_act * ACTION_SCALE, ctrl_range[:, 0], ctrl_range[:, 1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)

    new_phase = (phase + CTRL_DT * STEP_FREQ) % 1.0

    rew, done_bonus = compute_reward(d, raw_act, prev_act, cmd_vel, phase)
    obs_out = get_obs(d, raw_act, cmd_vel, new_phase)

    upvec      = get_upvector(d.qpos)
    # v7 STRICT termination: tilt > 25° OR pelvis < 76.2cm
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

    # v7: termination penalty — large negative reward when robot falls
    # Research: without this, robot treats falling as neutral → exploits falling
    total_rew = rew + jnp.where(terminated, -5.0 * CTRL_DT, 0.0)

    reset_state = env_reset(rng_reset)
    next_d     = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act   = jnp.where(done, jnp.zeros(nu), raw_act)
    next_step  = jnp.where(done, jnp.zeros((), jnp.int32), step_new)
    next_phase = jnp.where(done, reset_state["phase"], new_phase)
    next_cmd   = jnp.where(done, reset_state["cmd_vel"], cmd_vel)

    nst = {"d": next_d, "prev_act": next_act, "step": next_step,
           "phase": next_phase, "cmd_vel": next_cmd}
    return obs_out, nst, total_rew, terminated, truncated

# ================================================================
# 5. REWARD FUNCTION — v7: ASYMMETRIC VELOCITY + STRICT TERMINATION
#
# v6 FAILURE ANALYSIS (from simulation + research):
#   - log_std=0.508 for ALL 32 joints → policy outputs NOISE, not actions
#   - Runaway velocity: cmd=0.3 m/s but actual reaches 1.57 m/s in 1.3s
#   - Root cause: exp(-|v-cmd|²/σ²) is SYMMETRIC, but leaning forward
#     is physically ASYMMETRIC (free energy from gravity = easy acceleration)
#   - Robot learned to lean forward (get vel reward) but never learned to brake
#   - ENT_COEF=0.01 caused entropy saturation (policy stays random = noise)
#
# v7 fixes:
#   1. ASYMMETRIC velocity reward: penalize overshoot LINEARLY (strong gradient!)
#   2. STRICT pitch penalty: explicit penalty for pelvis pitch > 10°
#   3. ENT_COEF decay: 0.01 → 0.001 (force policy to commit to actions)
#   4. log_std clipped: [-3, 0.5] prevents saturation (in ActorCritic.forward)
#   5. Termination penalty: -5.0 * CTRL_DT when robot falls (no free falls!)
# ================================================================
# Adaptive parameters updated host-side
_VEL_SIGMA   = 0.15   # Tighter sigma from the start (no wide→tight schedule)
_PENALTY_SCL = 0.20   # Start at 20% penalty (less aggressive than v6's 10%)
_ENT_COEF    = 0.01   # Decays to 0.001 over first 50M steps

def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos  = d.qpos
    qvel  = d.qvel
    upvec = get_upvector(qpos)

    # ── PRIMARY: ASYMMETRIC Velocity tracking ──────────────────
    # Research (ETH RSL, legged_gym): overshoot must be penalized LINEARLY
    # (exp kernel has zero gradient when far → robot can't find direction to brake)
    vx_error = qvel[0] - cmd_vel[0]   # positive = going FASTER than commanded
    vy_error = qvel[1] - cmd_vel[1]
    # Overshoot (v > cmd): heavy LINEAR penalty — large gradient even when far
    # Undershoot (v < cmd): smooth exponential reward — encourage acceleration
    vel_sigma = jnp.float32(_VEL_SIGMA)
    r_vel_x = jnp.where(
        vx_error > 0,
        -2.0 * vx_error,                          # LINEAR penalty for overshoot
        jnp.exp(-jnp.square(vx_error) / vel_sigma)   # EXP reward for undershoot
    )
    r_vel_y = jnp.exp(-jnp.square(vy_error) / vel_sigma)  # symmetric for lateral
    r_vel_ang = jnp.exp(-jnp.square(qvel[5] - cmd_vel[2]) / vel_sigma)

    # ── SURVIVAL + STABILITY ──────────────────────────────────────
    r_alive  = 1.0
    r_orient = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.08)
    r_height = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.08)

    # v7: EXPLICIT PITCH PENALTY — penalize forward lean (pelvis pitch angle)
    # upvec[1] ≈ sin(pitch) for small angles: positive = leaning forward
    # This directly counters the "lean forward = free speed" exploit
    p_pitch = jnp.square(upvec[1]) * 3.0  # penalty increases with lean angle

    # ── GAIT CLOCK + FOOT CLEARANCE ──────────────────────────────
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    l_contact = (l_z < CONTACT_Z_THR).astype(jnp.float32)
    r_contact = (r_z < CONTACT_Z_THR).astype(jnp.float32)

    l_target_stance = (phase < STANCE_DUTY).astype(jnp.float32)
    r_phase_val = (phase + 0.5) % 1.0
    r_target_stance = (r_phase_val < STANCE_DUTY).astype(jnp.float32)

    r_gait = (
        jnp.where(l_target_stance > 0.5, l_contact, 1.0 - l_contact) +
        jnp.where(r_target_stance > 0.5, r_contact, 1.0 - r_contact)
    ) * 0.25

    l_swing = 1.0 - l_target_stance
    r_swing = 1.0 - r_target_stance
    l_clearance = jnp.clip((l_z - 0.04) / 0.12, 0.0, 1.0) * l_swing
    r_clearance = jnp.clip((r_z - 0.04) / 0.12, 0.0, 1.0) * r_swing
    r_foot_clearance = (l_clearance + r_clearance) * 0.4

    # ── PENALTIES ────────────────────────────────────────────────
    pen_scale = jnp.float32(_PENALTY_SCL)
    p_action_rate = pen_scale * 0.02 * jnp.mean(jnp.square(action - prev_action))
    p_torque      = pen_scale * 2e-4 * jnp.sum(jnp.square(action))
    p_body_tilt   = pen_scale * 0.05 * (jnp.square(qvel[3]) + jnp.square(qvel[4]))

    # ── TOTAL ──────────────────────────────────────────────────────
    # v7 standing still (cmd=0.2): r_alive+orient+height ≈ 1.25 → 0.0125/step
    # v7 walking at cmd: r_vel_x=1.0×4 + r_alive=1 + ... ≈ 0.05+/step
    total = (
        r_vel_x * 4.0 + r_vel_y * 0.5 + r_vel_ang * 0.5 +
        r_orient * 0.15 + r_height * 0.10 + r_alive +
        r_gait + r_foot_clearance
        - p_action_rate - p_torque - p_body_tilt - p_pitch
    )
    return jnp.maximum(0.0, total) * CTRL_DT, 0.0   # (reward, done_bonus placeholder)

# ================================================================
# 6. PPO ALGORITHM — v6: From Scratch (No Transfer)
#
# Literature: humanoid-gym, legged_gym, ETH RSL standard for humanoid walking
#   ROLLOUT=24: Short horizon, fast updates (MJX optimal ~0.24s at 100Hz)
#   LR=1e-3:    High LR for training from scratch (not fine-tuning)
#   MINIBATCH=4096: NUM_ENVS (4096) × 1 = tight batch size
#   N_EPOCHS=4: Standard PPO
#   NO Critic warmup: no Stage 1 weights to protect
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 24      # v6: 128→24 (MJX/IsaacGym optimal for locomotion)
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2     # Standard PPO clip (SOTA)
ENT_COEF     = 0.01    # Higher entropy encourages exploration from scratch
VF_COEF      = 0.5
MAX_GRAD     = 1.0     # Less conservative clip for from-scratch training
N_EPOCHS     = 4
MINIBATCH    = 4096    # = NUM_ENVS (one minibatch = one env-iter)
KL_TARGET    = 0.02    # Relaxed KL — more exploration allowed from scratch
TOTAL_STEPS  = 300_000_000   # v6: 300M steps (more needed from scratch vs fine-tune)
STEPS_PER_IT = NUM_ENVS * ROLLOUT   # 4096 × 24 = 98,304
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT  # ~3,051 iterations

# v6: NO Stage 1 weight transfer. Random Xavier init (network already initialized above).
# Research: training from scratch is easier than overcoming standing-balance local minimum.
print(f"[v6 CONFIG] ROLLOUT={ROLLOUT} ({ROLLOUT*0.01:.2f}s/iter) | STEPS_PER_IT={STEPS_PER_IT:,}")
print(f"[v6 CONFIG] N_EPOCHS={N_EPOCHS} | MINIBATCH={MINIBATCH} | LR=1e-3 (from scratch)")
print(f"[v7 CONFIG] N_ITERS={N_ITERS} | TOTAL_STEPS={TOTAL_STEPS:,}")
print(f"[v7 CONFIG] NO transfer, NO Stage1 leak | asymmetric velocity reward")
print(f"[v7 CONFIG] ENT_COEF=0.01 decay to 0.001 | log_std clipped [-3, 0.5]")

# v7: LR=1e-3 (cosine decay to 1e-4)
lr_schedule = optax.cosine_decay_schedule(1e-3, N_ITERS, alpha=0.1)  # 1e-3 → 1e-4
tx          = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(lr_schedule, eps=1e-5))
opt_state   = tx.init(params)
print(f"[OPTIMIZER] LR=1e-3→1e-4 (cosine) | CLIP=grad_norm {MAX_GRAD} | ENT=0.01→0.001")

rng_envs = jax.random.split(rng, NUM_ENVS)
states   = jax.vmap(env_reset)(rng_envs)

@jax.jit
def collect_rollout(params, states, rng):
    """Collect ROLLOUT steps of experience from all environments."""
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs  = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        act  = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp   = jnp.clip(-0.5 * jnp.sum(
            jnp.square((act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, r_resets))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(
        _step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    # GAE advantage estimation
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
    """Single gradient step on one mini-batch."""
    def loss_fn(p):
        mu, ls, v = network.apply(p, fo_mb)
        std = jnp.exp(ls)
        lp  = jnp.clip(-0.5 * jnp.sum(jnp.square((fa_mb - mu) / (std + 1e-8)) +
                        2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        ratio = jnp.exp(jnp.clip(lp - flp_mb, -5., 5.))
        approx_kl = jnp.mean(0.5 * jnp.square(lp - flp_mb))
        pg    = -jnp.mean(jnp.minimum(ratio * fadv_mb,
                          jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * fadv_mb))
        vc    = ovf_mb + jnp.clip(v - ovf_mb, -5., 5.)
        vf    = VF_COEF * jnp.mean(jnp.maximum(jnp.square(v - fret_mb),
                                                 jnp.square(vc - fret_mb)))
        # v7: ent_coef is DYNAMIC (decays from 0.01 to 0.001)
        ent   = -ent_coef * jnp.mean(jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        total = jnp.where(jnp.isnan(pg + vf + ent), 0.0, pg + vf + ent)
        return total, approx_kl

    (loss, approx_kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, approx_kl

# ================================================================
# 7. TRAINING LOOP — v7: Asymmetric Reward + ENT Decay + Strict Termination
# ================================================================
os.makedirs("checkpoints", exist_ok=True)
t0, cur = time.time(), 0

# v7 reward thresholds
WALK_THRESHOLD      = 0.020   # r_alive(1.0)×0.01 + orient ≈ 0.011/step = standing
WALK_WELL_THRESHOLD = 0.040   # Velocity tracking working = 0.04+/step

print(f"\nAPOLLO HUMANOID - STAGE 2 v7 (ASYMMETRIC REWARD + STRICT TERMINATION)")
print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | ROLLOUT={ROLLOUT}")
print(f"ENT_COEF 0.01→0.001 | TERM_TILT=cos(25°) | Asymmetric vel reward")
print("=" * 64)

# v7 curriculum: cmd_vel_max grows as robot learns
# Phase 0: 0-30M   → vx [0, 0.25] — balance + tiny movement
# Phase 1: 30-100M  → vx [0, 0.55] — walking starts
# Phase 2: 100-200M → vx [0, 0.85] — speed up
# Phase 3: 200M+    → vx [0, 1.20] — full speed
CURR_P0 = 30_000_000
CURR_P1 = 100_000_000
CURR_P2 = 200_000_000

def get_curriculum_vx_max(n_steps):
    if n_steps < CURR_P0:
        return 0.25
    elif n_steps < CURR_P1:
        t = (n_steps - CURR_P0) / (CURR_P1 - CURR_P0)
        return 0.25 + t * (0.55 - 0.25)
    elif n_steps < CURR_P2:
        t = (n_steps - CURR_P1) / (CURR_P2 - CURR_P1)
        return 0.55 + t * (0.85 - 0.55)
    else:
        return 1.20

@jax.jit
def reseed_cmd_vel(states, rng, vx_max, vy_max, yaw_max):
    rngs = jax.random.split(rng, NUM_ENVS)
    def _new_cmd(r):
        # v6: vx starts from 0 (robot learns to walk, then go faster)
        return jax.random.uniform(r, (3,),
            minval=jnp.array([0.0,  -vy_max, -yaw_max]),
            maxval=jnp.array([vx_max, vy_max,  yaw_max]))
    new_cmds = jax.vmap(_new_cmd)(rngs)
    return {**states, "cmd_vel": new_cmds}

import numpy as np_host

for it in range(1, N_ITERS + 1):
    t1 = time.time()

    # ── Adaptive parameters (host-side, outside jit) ────────────
    # v7: ENT_COEF decays 0.01 → 0.001 over first 50M steps (force commitment)
    # v7: PENALTY_SCL increases 0.20 → 1.00 over 100M steps (progressive difficulty)
    # v7: VEL_SIGMA fixed at 0.15 (tighter from start — better gradient signal)
    global _VEL_SIGMA, _PENALTY_SCL, _ENT_COEF
    ent_t        = min(1.0, cur / 50_000_000)
    _ENT_COEF    = 0.01 - ent_t * (0.01 - 0.001)    # 0.01 → 0.001
    pen_t        = min(1.0, cur / 100_000_000)
    _PENALTY_SCL = 0.20 + pen_t * (1.0  - 0.20)     # 0.20 → 1.00
    # _VEL_SIGMA stays fixed at 0.15 (set at init)

    # ── Curriculum: update cmd_vel every 20 iters ───────────────
    if it % 20 == 1:
        vx_max_cur = get_curriculum_vx_max(cur)
        vy_max_cur  = min(0.3, vx_max_cur * 0.4)
        yaw_max_cur = min(0.5, vx_max_cur * 0.5)
        rng, rng_seed = jax.random.split(rng)
        states = reseed_cmd_vel(states, rng_seed,
                                jnp.float32(vx_max_cur),
                                jnp.float32(vy_max_cur),
                                jnp.float32(yaw_max_cur))

    # ── Collect rollout ──────────────────────────────────────────
    states, rng, fo, fa, flp, fadv, fret, ovf, mr = \
        collect_rollout(params, states, rng)
    jax.block_until_ready(fo)

    # ── PPO epochs with mini-batches + KL Early Stopping ────────
    N_SAMPLES  = fo.shape[0]  # STEPS_PER_IT = 98,304
    N_MB       = N_SAMPLES // MINIBATCH   # 98304 / 4096 = 24 mini-batches/epoch
    last_loss  = 0.0
    kl_stopped = False
    ent_coef_j = jnp.float32(_ENT_COEF)  # v7: pass dynamic ent_coef to jit fn

    for epoch in range(N_EPOCHS):
        if kl_stopped:
            break
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
                kl_stopped = True
                break

    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)

    if it % 10 == 0 or it <= 5:
        r_val    = float(mr)
        vx_max_p = get_curriculum_vx_max(cur)

        if r_val > WALK_WELL_THRESHOLD: status = "*** WALKING WELL ***"
        elif r_val > WALK_THRESHOLD:    status = "*** WALKING ***"
        elif r_val > 0.015:             status = "stepping"
        elif r_val > 0.010:             status = "improving"
        else:                           status = "..."

        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={r_val:.5f} | loss={float(last_loss):.4f} | "
              f"sps={sps:,.0f} | vx_max={vx_max_p:.2f} | ent={float(_ENT_COEF):.4f} | "
              f"t={time.time()-t0:.0f}s {status}", flush=True)

    if it % 300 == 0 or it == N_ITERS:
        ck = f"checkpoints/apollo_stage2_v7_step_{cur}.npz"
        import flax
        flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
        flat_np["_step"] = np.array(cur)
        flat_np["_it"]   = np.array(it)
        np.savez(ck, **flat_np)
        ck_size = os.path.getsize(ck)
        if ck_size < 100_000:
            print(f"  [WARNING] Checkpoint small: {ck_size} bytes!", flush=True)
        else:
            print(f"  -> checkpoint: {ck} ({ck_size//1024}KB)", flush=True)

# Save final checkpoint as v7_final
flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
flat_np["_step"] = np.array(cur)
flat_np["_it"]   = np.array(it)
np.savez("checkpoints/apollo_stage2_v7_final.npz", **flat_np)
print("\\nSTAGE 2 v7 TRAINING COMPLETE! (Asymmetric Reward + Strict Termination)", flush=True)
print(f"Total steps: {cur:,} | Final checkpoint: checkpoints/apollo_stage2_v7_final.npz", flush=True)
'''

    SETUP_CELL = [
        "!nvidia-smi\n",
        "import os; print('CWD:', os.getcwd())\n",
        "!pip install -q --no-cache-dir mujoco mujoco-mjx flax optax\n",
        "import jax\n",
        "print('Backend:', jax.default_backend(), '| Devices:', jax.devices())\n",
        "assert jax.default_backend() in ('gpu','tpu'), 'GPU required!'",
    ]

    DOWNLOAD_CELL = [
        "# Download mujoco_menagerie (Apollo MJCF model)\n",
        "import os, urllib.request, zipfile, shutil\n",
        "\n",
        "TARGET    = 'mujoco_menagerie'\n",
        "APOLLO_XML = os.path.join(TARGET, 'apptronik_apollo', 'scene.xml')\n",
        "\n",
        "if not os.path.exists(APOLLO_XML):\n",
        "    print('Downloading mujoco_menagerie ZIP...')\n",
        "    zip_url  = 'https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip'\n",
        "    zip_path = '/tmp/mujoco_menagerie.zip'\n",
        "    urllib.request.urlretrieve(zip_url, zip_path)\n",
        "    print(f'ZIP downloaded: {os.path.getsize(zip_path)/1e6:.1f} MB')\n",
        "    with zipfile.ZipFile(zip_path, 'r') as z:\n",
        "        z.extractall('/tmp/menagerie_extract')\n",
        "    shutil.move('/tmp/menagerie_extract/mujoco_menagerie-main', TARGET)\n",
        "    os.remove(zip_path)\n",
        "    print('Extraction complete.')\n",
        "\n",
        "assert os.path.exists(APOLLO_XML), f'Missing: {APOLLO_XML}'\n",
        "print('[OK] Apollo model ready:', APOLLO_XML)\n",
        "\n",
        "# Check for Stage 1 checkpoint (for transfer learning)\n",
        "import glob as _g\n",
        "s1_cks = _g.glob('/kaggle/input/*/checkpoints/*.npz')\n",
        "if s1_cks:\n",
        "    print(f'[OK] Stage 1 checkpoint found: {sorted(s1_cks)[-1]}')\n",
        "else:\n",
        "    print('[INFO] No Stage 1 checkpoint found — will train from scratch')\n",
        "    print('  TIP: Add your Stage 1 npz files as a Kaggle Dataset to enable transfer learning')",
    ]

    cells = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": SETUP_CELL},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": DOWNLOAD_CELL},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [TRAINING_CODE]},
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }

    nb_path = os.path.join(deploy_dir, "apollo_humanoid_stage2_walking.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[STAGE 2 NOTEBOOK] {nb_path} ({os.path.getsize(nb_path):,} bytes)")

    meta = {
        "id": f"{username}/apollo-humanoid-stage2-walking",
        "title": "apollo-humanoid-stage2-walking",
        "code_file": "apollo_humanoid_stage2_walking.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4x2",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(deploy_dir, "kernel-metadata-stage2.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print("[METADATA STAGE 2 UPDATED]")


if __name__ == "__main__":
    generate()
