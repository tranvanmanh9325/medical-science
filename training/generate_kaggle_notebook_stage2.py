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
    # STAGE 2 v1 — WALKING & LOCOMOTION + PUSH RECOVERY
    #
    # Architecture decisions (expert-level):
    #   1. Transfer learning: Stage 1 weights (105-dim) → zero-pad → 114-dim
    #      New obs added: cmd_vel(3) + gait_phase(4) + foot_contact(2) = 9 dims
    #   2. Single training run, full velocity range [0, 0.8 m/s]
    #      Robot already balances from Stage 1 → velocity reward guides walking
    #   3. Progressive push recovery: starts at 5M steps, grows to 80N at 50M steps
    #   4. Gait Clock (CPG): 1.2 Hz alternating, sin/cos encoded → L/R synchronization
    #   5. Action scale: 0.25 (needs larger joint excursion than Stage 1's 0.1)
    #   6. Foot contact: site_xpos height threshold (vmap-compatible, no XML changes)
    #   7. EPISODE_LEN = 500 steps (5s at CTRL_DT=0.01s)
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
# Progressive random external forces applied to pelvis (base_link body_id=1)
PUSH_START_STEP = 500_000    # 5M env-steps before pushes start
PUSH_MAX_STEP   = 5_000_000  # 50M steps: pushes reach full magnitude
PUSH_MAX_FORCE  = 80.0       # N — ETH RSL standard for humanoid push robustness
PUSH_INTERVAL   = 200        # Apply push every 2s (200 ctrl steps)

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
# 5. REWARD FUNCTION — Stage 2 (Walking + Push Recovery)
#
# Philosophy: All-positive Gaussian core (inherited from Stage 1)
#             + velocity tracking (primary task)
#             + gait clock schedule (guide stepping pattern)
#             + energy penalties (sim-to-real smoothness)
# ================================================================
def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos = d.qpos
    qvel = d.qvel
    upvec = get_upvector(qpos)

    # ── PRIMARY TASK: Velocity tracking (most important signal) ──
    # Narrower kernel σ=0.30 (was σ=0.50): standing still at cmd=0.5m/s now gets
    # exp(-0.25/0.09)=0.06 vs 0.23 before — forces policy to actually move
    r_vel_lin = jnp.exp(-jnp.sum(jnp.square(qvel[:2] - cmd_vel[:2])) / 0.09)   # XY velocity
    r_vel_ang = jnp.exp(-jnp.square(qvel[5] - cmd_vel[2]) / 0.09)               # Yaw rate

    # ── STABILITY (secondary — allow some tilt during walking) ──
    r_orient  = jnp.exp(-jnp.sum(jnp.square(upvec[:2])) / 0.10)  # Softer than Stage 1 (0.05)
    r_height  = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.10) # Hips can dip during steps
    r_alive   = 0.2

    # ── GAIT CLOCK SCHEDULE REWARD ────────────────────────────────
    # Stance phase: left in stance when phase < STANCE_DUTY
    #               right in stance when (phase + 0.5) % 1.0 < STANCE_DUTY
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    l_contact = (l_z < CONTACT_Z_THR).astype(jnp.float32)
    r_contact = (r_z < CONTACT_Z_THR).astype(jnp.float32)

    # Reward 1 if foot matches target gait phase, 0 if not
    l_target_stance = (phase < STANCE_DUTY).astype(jnp.float32)
    r_phase = (phase + 0.5) % 1.0
    r_target_stance = (r_phase < STANCE_DUTY).astype(jnp.float32)

    r_gait = (
        jnp.where(l_target_stance > 0.5, l_contact, 1.0 - l_contact) +
        jnp.where(r_target_stance > 0.5, r_contact, 1.0 - r_contact)
    ) * 0.3   # 0.3 weight: guide but don't over-constrain natural gait

    # ── ENERGY & SMOOTHNESS PENALTIES ────────────────────────────
    p_action_rate = 0.02  * jnp.mean(jnp.square(action - prev_action))  # Anti-jitter
    p_torque      = 2e-4  * jnp.sum(jnp.square(action))                  # Energy efficiency
    p_body_tilt   = 0.05  * (jnp.square(qvel[3]) + jnp.square(qvel[4])) # Anti-wobble

    # ── TOTAL REWARD ─────────────────────────────────────────────
    # Weights: vel_lin=2.0 (primary), vel_ang=0.5, orient=0.5, height=0.3, alive=0.2, gait=0.6
    # Max reward when walking well: (2.0+0.5+0.5+0.3+0.2+0.6)*0.01 = 0.041/step
    # When cmd_vel=0 (standing): orientation+height rewards dominate → ~0.010/step
    total = (
        r_vel_lin * 2.0 + r_vel_ang * 0.5 +
        r_orient  * 0.5 + r_height  * 0.3 + r_alive +
        r_gait
        - p_action_rate - p_torque - p_body_tilt
    )
    return jnp.maximum(0.0, total) * CTRL_DT

# ================================================================
# 6. PPO ALGORITHM
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 64     # Longer than Stage 1 (32) → better credit assignment for locomotion
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.012  # Higher than original 0.005 — prevents entropy collapse (log_std → -3.0)
VF_COEF      = 0.5
MAX_GRAD     = 0.5
TOTAL_STEPS  = 150_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT  # 4096 * 64 = 262,144 per iteration
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT  # ~572 iterations

lr_schedule = optax.linear_schedule(2e-4, 1e-5, N_ITERS)  # Lower LR (fine-tuning)
tx          = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(lr_schedule, eps=1e-5))
opt_state   = tx.init(params)

rng_envs    = jax.random.split(rng, NUM_ENVS)
states      = jax.vmap(env_reset)(rng_envs)

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs  = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(st)
        mu, ls, val = network.apply(p, obs)
        std  = jnp.exp(ls)
        act  = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1., 1.)
        lp   = -0.5 * jnp.sum(
            jnp.square((act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, r_resets))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(
        _step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(fst)
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
        std  = jnp.exp(ls)
        lp   = -0.5 * jnp.sum(
            jnp.square((fa - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        lp   = jnp.clip(lp, -10.0, 10.0)
        lr_  = jnp.clip(lp - flp, -5.0, 5.0)
        ratio = jnp.exp(lr_)
        pg   = -jnp.mean(jnp.minimum(ratio * fadv,
                                      jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * fadv))
        vc   = ovf + jnp.clip(v - ovf, -5.0, 5.0)
        vf   = VF_COEF * jnp.mean(jnp.maximum(
            jnp.square(v - fret), jnp.square(vc - fret)))
        ent  = -ENT_COEF * jnp.mean(
            jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        return jnp.where(jnp.isnan(pg + vf + ent), 0.0, pg + vf + ent)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
    upd, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, upd)
    return params, opt_state, fst, rng, jnp.mean(rews), loss

# ================================================================
# 7. TRAINING LOOP — With Push Recovery Curriculum
#
# Push recovery is implemented via forced periodic random perturbations
# applied directly in the rollout (outside jit), starting at 5M steps.
# Magnitude scales linearly: 0N at 5M → 80N at 50M steps → constant 80N after.
# ================================================================
os.makedirs("checkpoints", exist_ok=True)
t0, cur = time.time(), 0
print(f"\nSteps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | ROLLOUT={ROLLOUT}")
print(f"Reward: target rew > 0.018 (walking) with narrow kernel | σ_vel=0.09")
print(f"Transfer: {'YES (Stage 1 weights loaded)' if stage1_ck else 'NO (scratch)'}")
print(f"Push recovery curriculum: starts at {PUSH_START_STEP:,} env-steps")
print(f"Velocity curriculum: Phase1 [0,0.15]→Phase2 [0,0.45]→Phase3 [0,0.80] m/s")
print("=" * 64)

# ── Curriculum velocity boundaries (updated outside jit) ──────────
# Passed into env_reset via state re-seeding at episode end.
# We periodically re-seed HALF of envs to inject new velocity commands
# matching the current curriculum phase. The other half keeps existing state.
CURRICULUM_P1_END = 50_000_000   # 50M steps: vx max goes 0.15 → 0.45
CURRICULUM_P2_END = 120_000_000  # 120M steps: vx max goes 0.45 → 0.80

def get_curriculum_cmd_max(n_steps):
    """Return (vx_max, vy_max, yaw_max) based on current training steps."""
    if n_steps < CURRICULUM_P1_END:
        vx = 0.15
    elif n_steps < CURRICULUM_P2_END:
        # Linear interpolation 0.15 → 0.80 over 50M-120M steps
        t = (n_steps - CURRICULUM_P1_END) / (CURRICULUM_P2_END - CURRICULUM_P1_END)
        vx = 0.15 + t * (0.80 - 0.15)
    else:
        vx = 0.80
    vy  = min(0.3, vx * 0.4)
    yaw = min(0.4, vx * 0.5)
    return vx, vy, yaw

@jax.jit
def reseed_cmd_vel(states, rng, vx_max, vy_max, yaw_max):
    """Reseed cmd_vel for ALL envs with current curriculum velocity range."""
    rngs = jax.random.split(rng, NUM_ENVS)
    def _new_cmd(r):
        return jax.random.uniform(r, (3,),
            minval=jnp.array([0.0, -vy_max, -yaw_max]),
            maxval=jnp.array([vx_max, vy_max, yaw_max]))
    new_cmds = jax.vmap(_new_cmd)(rngs)
    return {**states, "cmd_vel": new_cmds}

for it in range(1, N_ITERS + 1):
    t1 = time.time()

    # ── Curriculum: update cmd_vel every 10 iters (≈2.6M steps) ──
    if it % 10 == 1:
        vx_max, vy_max, yaw_max = get_curriculum_cmd_max(cur)
        rng, rng_seed = jax.random.split(rng)
        states = reseed_cmd_vel(states, rng_seed,
                                jnp.float32(vx_max),
                                jnp.float32(vy_max),
                                jnp.float32(yaw_max))

    params, opt_state, states, rng, mr, loss = \
        ppo_iter(params, opt_state, states, rng)

    # block_until_ready ensures JAX async dispatch completes before measuring time
    jax.block_until_ready(params)
    cur += STEPS_PER_IT
    sps = STEPS_PER_IT / max(1e-5, time.time() - t1)

    if it % 5 == 0 or it == 1:
        r_val = float(mr)
        vx_max, _, _ = get_curriculum_cmd_max(cur)
        push_frac = min(1.0, max(0.0, (cur - PUSH_START_STEP) / (PUSH_MAX_STEP - PUSH_START_STEP)))
        push_mag  = PUSH_MAX_FORCE * push_frac if cur >= PUSH_START_STEP else 0.0

        # Thresholds recalibrated for narrow kernel (σ=0.09):
        # Standing still: ~0.008/step | Walking well: >0.018/step
        if r_val > 0.022:    status = "*** WALKING WELL ***"
        elif r_val > 0.016:  status = "*** WALKING ***"
        elif r_val > 0.010:  status = "stepping"
        elif r_val > 0.006:  status = "improving"
        else:                status = "..."

        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | "
              f"rew={r_val:.5f} | loss={float(loss):.4f} | "
              f"sps={sps:,.0f} | push={push_mag:.0f}N | "
              f"vx_max={vx_max:.2f} | t={time.time()-t0:.0f}s {status}", flush=True)

    if it % 50 == 0 or it == N_ITERS:
        ck = f"checkpoints/apollo_stage2_v2_step_{cur}.npz"
        # ── CRITICAL FIX: jnp.savez is async and produces corrupt/empty files ──
        # Must convert to numpy AFTER block_until_ready, then use np.savez
        flat = flax.traverse_util.flatten_dict(params, sep="/")
        flat_np = {k: np.array(v) for k, v in flat.items()}
        np.savez(ck, **flat_np)
        ck_size = os.path.getsize(ck)
        if ck_size < 100_000:  # Healthy checkpoint should be >800KB
            print(f"  [WARNING] Checkpoint suspiciously small: {ck_size} bytes — possible save error!")
        else:
            print(f"  -> checkpoint: {ck} ({ck_size//1024}KB)", flush=True)

print("STAGE 2 v2 COMPLETE", flush=True)
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
