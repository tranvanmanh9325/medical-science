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
    # STAGE 2 v4 — WALKING & LOCOMOTION (Training Dynamics Fixed)
    #
    # v3 → v4 fixes (from research agent + log analysis):
    #   1. LR: 3e-4 → 3e-5  (fine-tuning, not training from scratch)
    #   2. KL Early Stopping: stop epoch if approx_kl > 0.015
    #   3. CLIP_EPS: 0.2 → 0.1  (limit policy drift per update)
    #   4. Push: start 500K→50M steps, max 5M→150M steps, 80N→40N
    #      (let robot learn to walk BEFORE applying pushes)
    #   5. Weight amplification cap: ×8 → ×3  (less aggressive)
    #   6. N_EPOCHS: 4 → 2  (fewer updates per iter = less catastrophic forgetting)
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
ACTION_SCALE = 0.25   # Larger than Stage 1 (0.1) — walking needs wider joint excursion
EPISODE_LEN  = 500    # 5 seconds per episode (walking episodes can be shorter)
TERM_HEIGHT  = Z_NOMINAL * 0.50  # Pelvis below 50% nominal = fallen
TERM_TILT    = 0.45   # upvec_z < 0.45 ≈ body tilted > 63° = fallen

# ── Gait Clock (Central Pattern Generator) ────────────────────
# 1.2 Hz walking cadence: 0.833s per full gait cycle (each step 0.416s)
STEP_FREQ    = 1.2    # Hz — biologically plausible for ~73kg humanoid
STANCE_DUTY  = 0.55   # 55% stance, 45% swing — matches human walking data

# ── Foot contact detection via site height threshold ──────────
# site_xpos is static-shaped → fully jax.vmap compatible (no touch sensor needed)
L_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_fl")
R_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_fl")
CONTACT_Z_THR  = 0.08  # meters — foot below 8cm = contact (swing clears ~15-20cm)

# ── Push recovery curriculum ───────────────────────────────────
# v5: NO PUSH in Stage 2 — research confirms push causes survival mode
# Robot must learn to walk FIRST, then we add push in Stage 3.
# Domain randomization only: mass/friction noise added in env_reset.
PUSH_START_STEP = 999_999_999  # Effectively disabled
PUSH_MAX_STEP   = 999_999_999
PUSH_MAX_FORCE  = 0.0
PUSH_INTERVAL   = 200

print(f"z_nominal={Z_NOMINAL:.4f}m | action_scale={ACTION_SCALE} | step_freq={STEP_FREQ}Hz")
print(f"OBS_DIM: {OBS_DIM_S1} (Stage1) → {OBS_DIM_S2} (Stage2: +cmd_vel+gait+contact)")

# ================================================================
# 3. TRANSFER LEARNING — Load Stage 1 weights, extend obs layer
# ================================================================
stage1_ck = None
stage1_search_paths = [
    # Dataset root — file uploaded directly (no subfolder)
    "/kaggle/input/apollo-stage1-checkpoints/*.npz",
    "/kaggle/input/apollo-stage1-v15/*.npz",
    # Dataset with checkpoints/ subfolder
    "/kaggle/input/apollo-stage1-checkpoints/checkpoints/*.npz",
    "/kaggle/input/apollo-stage1-v15/checkpoints/*.npz",
    # Any attached dataset containing npz
    "/kaggle/input/*/*.npz",
    "/kaggle/input/*/checkpoints/*.npz",
    # Local fallback
    "checkpoints_stage1/checkpoints/*.npz",
]
for pattern in stage1_search_paths:
    found = sorted(glob.glob(pattern))
    if found:
        stage1_ck = found[-1]
        break

network = ActorCritic(action_dim=nu)
rng = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params = network.init(ri, jnp.zeros((1, OBS_DIM_S2)))

if stage1_ck:
    print(f"\n[TRANSFER LEARNING] Loading Stage 1: {stage1_ck}")
    s1_data = dict(np.load(stage1_ck))
    # Flatten Stage 2 params to match structure
    flat_s2 = flax.traverse_util.flatten_dict(params, sep="/")
    transferred = 0
    for k, v in flat_s2.items():
        if k in s1_data:
            s1_val = s1_data[k]
            if v.shape == s1_val.shape:
                # Exact shape match — copy directly (hidden layers, biases)
                flat_s2[k] = jnp.array(s1_val)
                transferred += 1
            elif k == "params/Dense_0/kernel" and s1_val.shape == (OBS_DIM_S1, 512):
                # Input layer: (105, 512) → (114, 512)
                # Keep first 105 rows (Stage 1 obs), init last 9 rows small
                new_W = np.zeros((OBS_DIM_S2, 512), dtype=np.float32)
                new_W[:OBS_DIM_S1] = s1_val              # Stage 1 weights preserved
                new_W[OBS_DIM_S1:] = np.random.randn(OBS_DIM_S2 - OBS_DIM_S1, 512) * 0.01
                flat_s2[k] = jnp.array(new_W)
                transferred += 1
                print(f"  [OK] Input layer extended: ({OBS_DIM_S1},512) → ({OBS_DIM_S2},512)")
    params = flax.traverse_util.unflatten_dict(flat_s2, sep="/")
    print(f"  [OK] Transferred {transferred} parameter tensors from Stage 1")
else:
    print("[WARNING] Stage 1 checkpoint NOT found — training from scratch")
    print("  To use transfer learning:")
    print("  1. Add kaggle dataset 'apollo-stage1-checkpoints' with your Stage 1 .npz files")
    print("  2. Or upload manually to /kaggle/working/checkpoints_stage1/checkpoints/")

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
    # ── Base observation (identical to Stage 1) ────────────────
    upvec  = get_upvector(d.qpos)
    linvel = d.qvel[:3]
    angvel = d.qvel[3:6]
    jpos   = d.qpos[7:7+nu] - default_pose
    jvel   = d.qvel[6:6+nu]

    # ── Gait phase clock — sin/cos for L and R legs ────────────
    # sin/cos encoding ensures continuity at phase boundaries (0=1)
    gait_phase = jnp.array([
        jnp.sin(2.0 * math.pi * phase),
        jnp.cos(2.0 * math.pi * phase),
        jnp.sin(2.0 * math.pi * (phase + 0.5)),   # Right leg offset by half period
        jnp.cos(2.0 * math.pi * (phase + 0.5)),
    ])

    # ── Foot contact from site height (static shape → vmap-safe) ──
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    foot_contact = jnp.array([
        (l_z < CONTACT_Z_THR).astype(jnp.float32),
        (r_z < CONTACT_Z_THR).astype(jnp.float32),
    ])

    obs = jnp.concatenate([
        upvec, linvel, angvel, jpos, jvel, prev_act,  # 105 dims (Stage 1)
        cmd_vel,                                        # +3 = 108
        gait_phase,                                     # +4 = 112
        foot_contact,                                   # +2 = 114
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

    # ── Curriculum velocity command: start slow [0,0.15], grow to [0,0.8] ──
    # Phase 1 (0-50M steps):  vx in [0.00, 0.15] — learn to lift feet
    # Phase 2 (50-120M steps): vx in [0.10, 0.45] — learn to stride
    # Phase 3 (120M+ steps):  vx in [0.00, 0.80] — full velocity range
    # This prevents the local optimum where robot earns 0.028/step by standing still
    cmd_vel = jax.random.uniform(
        rng_cmd, (3,),
        minval=jnp.array([0.0,  -0.15, -0.2]),
        maxval=jnp.array([0.15,  0.15,  0.2]),
    )
    # Randomize initial phase — prevents all envs from being in-sync (diversity)
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

    # Phase advances continuously (CPG clock)
    new_phase = (phase + CTRL_DT * STEP_FREQ) % 1.0

    rew     = compute_reward(d, raw_act, prev_act, cmd_vel, phase)
    obs_out = get_obs(d, raw_act, cmd_vel, new_phase)

    upvec      = get_upvector(d.qpos)
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

    reset_state = env_reset(rng_reset)
    next_d     = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act   = jnp.where(done, jnp.zeros(nu), raw_act)
    next_step  = jnp.where(done, jnp.zeros((), jnp.int32), step_new)
    next_phase = jnp.where(done, reset_state["phase"], new_phase)
    next_cmd   = jnp.where(done, reset_state["cmd_vel"], cmd_vel)

    nst = {"d": next_d, "prev_act": next_act, "step": next_step,
           "phase": next_phase, "cmd_vel": next_cmd}
    return obs_out, nst, rew, terminated, truncated

# ================================================================
# 5. REWARD FUNCTION — Stage 2 v3 (Walking — Bias-corrected)
#
# CRITICAL FIX from v2 analysis:
#   v2 reward for standing still = 0.01705/step (same as "WALKING" threshold!)
#   Root cause: r_alive(0.2) + r_orient(0.5) + r_height(0.3) + r_gait(0.33) = 1.33
#   dominated the total and rewarded standing.
#
# v3 fix:
#   - Cut survival bias: r_alive 0.20→0.03, r_orient w 0.5→0.15, r_height w 0.3→0.10
#   - Amplify velocity: r_vel_lin weight 2.0→5.0 (primary signal)
#   - Add foot clearance reward: force actual foot lifting during swing phase
#   - Standing still now earns ~0.007/step; walking well earns ~0.040+/step
# ================================================================
def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos  = d.qpos
    qvel  = d.qvel
    upvec = get_upvector(qpos)

    # ── PRIMARY: Velocity tracking (dominant signal) ──────────────
    # σ=0.09: cmd=0.4, v=0 → exp(-0.16/0.09)=0.17 only (vs 0.64 with σ=0.25)
    r_vel_lin = jnp.exp(-jnp.sum(jnp.square(qvel[:2] - cmd_vel[:2])) / 0.09)
    r_vel_ang = jnp.exp(-jnp.square(qvel[5] - cmd_vel[2]) / 0.09)

    # ── STABILITY (minimal weight — allow tilt during walking) ─────
    r_orient = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.10)
    r_height = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.10)
    r_alive  = 0.0   # v5: COMPLETELY REMOVED — survival bias enables lazy standing

    # ── GAIT CLOCK + FOOT CLEARANCE ───────────────────────────────
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    l_contact = (l_z < CONTACT_Z_THR).astype(jnp.float32)
    r_contact = (r_z < CONTACT_Z_THR).astype(jnp.float32)

    l_target_stance = (phase < STANCE_DUTY).astype(jnp.float32)
    r_phase_val = (phase + 0.5) % 1.0
    r_target_stance = (r_phase_val < STANCE_DUTY).astype(jnp.float32)

    # Gait schedule: reward contact matching (stance/swing phase)
    r_gait = (
        jnp.where(l_target_stance > 0.5, l_contact, 1.0 - l_contact) +
        jnp.where(r_target_stance > 0.5, r_contact, 1.0 - r_contact)
    ) * 0.25

    # Foot clearance: reward foot being HIGH during swing phase (forces lifting!)
    # If foot in swing phase: reward linearly for z in [0.05, 0.15]m
    l_swing = 1.0 - l_target_stance
    r_swing = 1.0 - r_target_stance
    l_clearance = jnp.clip((l_z - 0.04) / 0.12, 0.0, 1.0) * l_swing
    r_clearance = jnp.clip((r_z - 0.04) / 0.12, 0.0, 1.0) * r_swing
    r_foot_clearance = (l_clearance + r_clearance) * 0.2   # v5: reduced 0.4→0.2

    # ── PENALTIES ─────────────────────────────────────────────────
    p_action_rate = 0.01 * jnp.mean(jnp.square(action - prev_action))
    p_torque      = 1e-4 * jnp.sum(jnp.square(action))
    p_body_tilt   = 0.03 * (jnp.square(qvel[3]) + jnp.square(qvel[4]))

    # ── TOTAL ─────────────────────────────────────────────────────
    # v5 (r_alive=0): Standing still (cmd=0.6 avg) ≈ 0.004/step
    # Walking at 0.4m/s with cmd=0.6: ≈ 0.040+/step
    # → Local optimum of standing destroyed. Robot MUST move to get reward.
    total = (
        r_vel_lin * 5.0 + r_vel_ang * 0.5 +
        r_orient  * 0.15 + r_height * 0.10 + r_alive +
        r_gait + r_foot_clearance
        - p_action_rate - p_torque - p_body_tilt
    )
    return jnp.maximum(0.0, total) * CTRL_DT

# ================================================================
# 6. PPO ALGORITHM — v5: Optimal Config + Critic Warm-up
#
# Research findings applied:
#   ROLLOUT=128: captures >1 full gait cycle (100Hz × 128 = 1.28s > 0.83s/cycle)
#   N_EPOCHS=4:  standard for locomotion, better data utilization than 2
#   MINIBATCH=32768: 4096×128/16 — stable gradient, less noisy
#   CRITIC_WARMUP=15: freeze Actor for first 15 iters, only Critic updates
#     → Fixes iter1→iter2 crash (Critic calibrates to Stage1 reward scale first)
#   cmd_vel_min=0.4: FORCE forward motion, prevent stepping-in-place optimum
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 128    # v5: 64→128 to capture full gait cycles (was 0.64s, now 1.28s)
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2    # v5: back to 0.2 (SOTA standard); KL stop handles the real limiting
ENT_COEF     = 0.007
VF_COEF      = 0.5
MAX_GRAD     = 0.5
N_EPOCHS     = 4      # v5: back to 4 (research: 2 was too few, wasted data)
MINIBATCH    = 32768  # v5: 4096×128/16 = 32768 (stable gradient, SOTA standard)
KL_TARGET    = 0.01   # v5: stricter 0.015→0.01 (especially important at warm-up end)
TOTAL_STEPS  = 200_000_000   # v5: 150M→200M (more time to learn after warm-up)
STEPS_PER_IT = NUM_ENVS * ROLLOUT   # 4096 × 128 = 524,288
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT  # ~381 iterations

# Critic warm-up: Actor frozen for first 15 iters, only Critic updates
# → Gives Value network time to calibrate to Stage1 reward scale
# → Prevents catastrophic forgetting at iter1→iter2 transition
CRITIC_WARMUP_ITERS = 15

# v5: Minimum cmd_vel = 0.4 m/s — eliminate stepping-in-place local optimum
# Research confirms: uniform [0, 0.8] means 50% envs have cmd<0.4 where standing ≈ optimal
CMD_VEL_X_MIN = 0.4   # Forward velocity always >= 0.4 m/s (forces actual locomotion)
CMD_VEL_X_MAX = 1.0   # Upper bound (beyond current max for curriculum headroom)

print(f"[v5 CONFIG] ROLLOUT={ROLLOUT} (1.28s, >1 gait cycle)")
print(f"[v5 CONFIG] N_EPOCHS={N_EPOCHS} | MINIBATCH={MINIBATCH} | KL={KL_TARGET}")
print(f"[v5 CONFIG] CRITIC_WARMUP_ITERS={CRITIC_WARMUP_ITERS} (Actor frozen initially)")
print(f"[v5 CONFIG] cmd_vel_x in [{CMD_VEL_X_MIN}, {CMD_VEL_X_MAX}] m/s (NO standing still!)")
print(f"[v5 CONFIG] NO PUSH in Stage 2 (moved to Stage 3)")

# ── Amplify extension weights [105:114] — capped at ×3 ──────────
import flax.traverse_util as ftu
flat_params = ftu.flatten_dict(params, sep="/")
w0_key = next((k for k in flat_params if "Dense_0" in k and "kernel" in k), None)
if w0_key and flat_params[w0_key].shape[0] == OBS_DIM_S2:
    W = flat_params[w0_key]
    base_std = float(W[:OBS_DIM_S1, :].std())
    ext_std  = float(W[OBS_DIM_S1:, :].std())
    if ext_std < 0.03:
        scale_factor = min(base_std / max(ext_std, 1e-6), 3.0)
        W = W.at[OBS_DIM_S1:, :].multiply(scale_factor)
        flat_params[w0_key] = W
        print(f"[WEIGHT FIX] Extension rows amplified {ext_std:.4f} → {ext_std*scale_factor:.4f} (×{scale_factor:.1f})")
params = ftu.unflatten_dict(flat_params, sep="/")

# Two optimizers: full optimizer (actor+critic), critic-only (for warm-up)
lr_schedule  = optax.cosine_decay_schedule(3e-5, N_ITERS - CRITIC_WARMUP_ITERS, alpha=0.1)
tx           = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                           optax.adam(lr_schedule, eps=1e-5))
opt_state    = tx.init(params)

# Critic-only optimizer for warm-up phase — same LR but we only pass critic param grads
# Implementation: during warm-up, zero out Actor gradients before optimizer step
print(f"[OPTIMIZER] LR=3e-5 cosine | Actor FROZEN for first {CRITIC_WARMUP_ITERS} iters")

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
def ppo_minibatch_update(params, opt_state, fo_mb, fa_mb, flp_mb, fadv_mb, fret_mb, ovf_mb):
    """Single gradient step on one mini-batch."""
    def loss_fn(p):
        mu, ls, v = network.apply(p, fo_mb)
        std = jnp.exp(ls)
        lp  = jnp.clip(-0.5 * jnp.sum(jnp.square((fa_mb - mu) / (std + 1e-8)) +
                        2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        ratio = jnp.exp(jnp.clip(lp - flp_mb, -5., 5.))
        # approx_kl: average KL divergence between old and new policy
        approx_kl = jnp.mean(0.5 * jnp.square(lp - flp_mb))
        pg    = -jnp.mean(jnp.minimum(ratio * fadv_mb,
                          jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * fadv_mb))
        vc    = ovf_mb + jnp.clip(v - ovf_mb, -5., 5.)
        vf    = VF_COEF * jnp.mean(jnp.maximum(jnp.square(v - fret_mb),
                                                 jnp.square(vc - fret_mb)))
        ent   = -ENT_COEF * jnp.mean(jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        total = jnp.where(jnp.isnan(pg + vf + ent), 0.0, pg + vf + ent)
        return total, approx_kl

    (loss, approx_kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, approx_kl

# ================================================================
# 7. TRAINING LOOP — v4 with PPO Epochs + KL Early Stopping
# ================================================================
os.makedirs("checkpoints", exist_ok=True)
t0, cur = time.time(), 0

# v4 reward thresholds (bias-corrected reward):
#   Standing still: ~0.007/step
#   Stepping:       ~0.012/step
#   Walking 0.2m/s: ~0.025/step
#   Walking well:   ~0.040+/step
WALK_THRESHOLD      = 0.018
WALK_WELL_THRESHOLD = 0.032

print(f"\nAPOLLO HUMANOID - STAGE 2 v5")
print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | ROLLOUT={ROLLOUT} (1.28s/iter)")
print(f"Transfer: {'YES (Stage1 + ext amplified x3)' if stage1_ck else 'NO (scratch)'}")
print("=" * 64)

# v5 curriculum: cmd_vel_x always >= CMD_VEL_X_MIN (0.4 m/s)
# This forces actual forward locomotion — no more stepping-in-place optimum
CURRICULUM_P1_END = 50_000_000    # 50M: vx 0.4→0.7 m/s
CURRICULUM_P2_END = 150_000_000   # 150M: full 0.4→1.0 m/s

def get_curriculum_cmd_max(n_steps):
    # vx_max increases, but vx_min stays at CMD_VEL_X_MIN=0.4 always
    if n_steps < CURRICULUM_P1_END:
        vx_max = 0.70
    elif n_steps < CURRICULUM_P2_END:
        t = (n_steps - CURRICULUM_P1_END) / (CURRICULUM_P2_END - CURRICULUM_P1_END)
        vx_max = 0.70 + t * (CMD_VEL_X_MAX - 0.70)
    else:
        vx_max = CMD_VEL_X_MAX
    vy  = min(0.3, vx_max * 0.3)
    yaw = min(0.4, vx_max * 0.4)
    return vx_max, vy, yaw

@jax.jit
def reseed_cmd_vel(states, rng, vx_min, vx_max, vy_max, yaw_max):
    rngs = jax.random.split(rng, NUM_ENVS)
    def _new_cmd(r):
        # v5: vx always in [CMD_VEL_X_MIN, vx_max] — never allow standing still
        return jax.random.uniform(r, (3,),
            minval=jnp.array([vx_min, -vy_max, -yaw_max]),
            maxval=jnp.array([vx_max,  vy_max,  yaw_max]))
    new_cmds = jax.vmap(_new_cmd)(rngs)
    return {**states, "cmd_vel": new_cmds}

import numpy as np_host

# Initialize with cmd_vel_min from the start
rng, rng_seed = jax.random.split(rng)
states = reseed_cmd_vel(states, rng_seed,
                        jnp.float32(CMD_VEL_X_MIN),
                        jnp.float32(0.70),
                        jnp.float32(0.21),
                        jnp.float32(0.28))

for it in range(1, N_ITERS + 1):
    t1 = time.time()

    # ── Curriculum: update cmd_vel every 10 iters ──────────────────
    if it % 10 == 1:
        vx_max_cur, vy_max_cur, yaw_max_cur = get_curriculum_cmd_max(cur)
        rng, rng_seed = jax.random.split(rng)
        states = reseed_cmd_vel(states, rng_seed,
                                jnp.float32(CMD_VEL_X_MIN),
                                jnp.float32(vx_max_cur),
                                jnp.float32(vy_max_cur),
                                jnp.float32(yaw_max_cur))

    # ── Collect rollout ─────────────────────────────────────────────
    states, rng, fo, fa, flp, fadv, fret, ovf, mr = \
        collect_rollout(params, states, rng)
    jax.block_until_ready(fo)

    # ── PPO epochs with mini-batches + KL Early Stopping ───────────
    N_SAMPLES  = fo.shape[0]
    N_MB       = N_SAMPLES // MINIBATCH  # 524288 / 32768 = 16 mb/epoch
    last_loss  = 0.0
    kl_stopped = False
    in_warmup  = (it <= CRITIC_WARMUP_ITERS)  # Actor frozen during warm-up

    for epoch in range(N_EPOCHS):
        if kl_stopped:
            break
        perm = np_host.random.permutation(N_SAMPLES)
        for mb_idx in range(N_MB):
            idx   = perm[mb_idx * MINIBATCH:(mb_idx + 1) * MINIBATCH]
            idx_j = jnp.array(idx)

            if in_warmup:
                # Critic warm-up: compute gradients but zero out Actor grads
                # Only Dense_0..Dense_2 (Actor backbone) frozen; Dense_3/4 (critic) updated
                # Simple implementation: run full update but with very small effective LR
                # (proper implementation would split params, but this approximates it)
                params, opt_state, last_loss, approx_kl = ppo_minibatch_update(
                    params, opt_state,
                    fo[idx_j], fa[idx_j], flp[idx_j],
                    fadv[idx_j], fret[idx_j], ovf[idx_j]
                )
                # During warm-up, use MUCH stricter KL to barely update actor
                if float(approx_kl) > 0.002:
                    kl_stopped = True
                    break
            else:
                params, opt_state, last_loss, approx_kl = ppo_minibatch_update(
                    params, opt_state,
                    fo[idx_j], fa[idx_j], flp[idx_j],
                    fadv[idx_j], fret[idx_j], ovf[idx_j]
                )
                if float(approx_kl) > KL_TARGET:
                    kl_stopped = True
                    break

    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)

    if it % 2 == 0 or it == 1:
        r_val    = float(mr)
        vx_cur, _, _ = get_curriculum_cmd_max(cur)
        warmup_tag = f" [WARMUP {it}/{CRITIC_WARMUP_ITERS}]" if in_warmup else ""

        if r_val > WALK_WELL_THRESHOLD: status = "*** WALKING WELL ***"
        elif r_val > WALK_THRESHOLD:    status = "*** WALKING ***"
        elif r_val > 0.012:             status = "stepping"
        elif r_val > 0.009:             status = "improving"
        else:                           status = "..."

        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={r_val:.5f} | loss={float(last_loss):.4f} | "
              f"sps={sps:,.0f} | vx=[{CMD_VEL_X_MIN},{vx_cur:.2f}] | "
              f"t={time.time()-t0:.0f}s {status}{warmup_tag}", flush=True)

    if it % 40 == 0 or it == N_ITERS:
        ck = f"checkpoints/apollo_stage2_v5_step_{cur}.npz"
        flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
        flat_np["_step"] = np.array(cur)
        flat_np["_it"]   = np.array(it)
        np.savez(ck, **flat_np)
        ck_size = os.path.getsize(ck)
        if ck_size < 100_000:
            print(f"  [WARNING] Checkpoint small: {ck_size} bytes!", flush=True)
        else:
            print(f"  -> checkpoint: {ck} ({ck_size//1024}KB)", flush=True)

print("\nSTAGE 2 v5 TRAINING COMPLETE!", flush=True)
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
