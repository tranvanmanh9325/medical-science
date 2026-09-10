import sys, argparse
import os, time, math, glob
import jax, jax.numpy as jnp
import optax, flax, flax.linen as nn
import mujoco, flax.traverse_util
from mujoco import mjx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .npz file to resume from")
args = parser.parse_args()

print("=" * 64)
print("  APOLLO HUMANOID - STAGE 2: NATURAL STAND & LOCOMOTION (v9)")
print("  112-dim Base Frame Obs | Stand-Still Reward | Linear Actor")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() in ("gpu", "tpu"), "GPU required!"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true --xla_gpu_enable_latency_hiding_scheduler=true")

OBS_DIM_S2 = 112
CKPT_DIR   = "/content/checkpoints"

# ================================================================
# 1. ACTOR-CRITIC — Modern Unbounded Gaussian & Linear Actor
# ================================================================
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in (512, 256, 128):
            x = nn.elu(nn.Dense(h)(x))
        # Linear actor output: ensures gradients always flow without tanh saturation
        mean = nn.Dense(self.action_dim)(x)
        # Sigmoid-bounded log_std: range [-4, -1], std in [0.018, 0.368]
        ls_param = self.param("log_std", nn.initializers.constant(0.0), (self.action_dim,))
        log_std  = -4.0 + 3.0 * nn.sigmoid(ls_param)
        value    = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value

# ================================================================
# 2. PHYSICS MODEL & CONFIGURATION
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
ctrl_range = jnp.array(mj_model.actuator_ctrlrange)

key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id < 0: key_id = 0
default_qpos = jnp.array(mj_model.key_qpos[key_id])
default_ctrl = jnp.array(mj_model.key_qpos[key_id][7:])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL    = float(default_qpos[2])       # ~1.016m standing height
ACTION_SCALE = 0.15                         # Smooth 0.15 rad (8.6 deg) control
EPISODE_LEN  = 500                          # 5s per episode at 100Hz
TERM_HEIGHT  = Z_NOMINAL * 0.75             # 0.762m
TERM_TILT    = -0.85                        # cos(31.8 deg) projection along gravity (-z)

STEP_FREQ   = 1.2                           # Gait frequency (Hz)
STANCE_DUTY = 0.55                          # Stance ratio per foot

L_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_fl")
R_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_fl")
CONTACT_Z_THR  = 0.05

print(f"z_nominal={Z_NOMINAL:.4f}m | TERM_HEIGHT={TERM_HEIGHT:.3f}m | ACTION_SCALE={ACTION_SCALE}")
print(f"OBS_DIM={OBS_DIM_S2} | CTRL_DT={CTRL_DT}s | step_freq={STEP_FREQ}Hz")

# ================================================================
# 3. QUATERNION MATH & OBSERVATION (Local Base Frame)
# ================================================================
def quat_rotate_inverse(q, v):
    """Rotates a 3D vector v by the inverse of quaternion q (MuJoCo format [w, x, y, z])."""
    q_w = q[0]
    q_vec = -q[1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * jnp.dot(q_vec, v) * 2.0
    return a + b + c

def get_obs(d, prev_act, cmd_vel, phase):
    """
    112-dim observation space in robot's local base coordinate frame:
      - projected_gravity (3): [0, 0, -1] rotated into base frame
      - base_lin_vel (3): linear velocity in base frame
      - base_ang_vel (3): angular velocity in base frame
      - joint_pos (32): joint angles relative to default standing pose
      - joint_vel (32): joint angular velocities
      - prev_act (32): previous network action
      - cmd_vel (3): commanded velocities [vx, vy, yaw_rate]
      - gait_phase (4): [sin(phi_L), cos(phi_L), sin(phi_R), cos(phi_R)]
    """
    base_quat = d.qpos[3:7]
    proj_grav = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))
    base_lin_vel = quat_rotate_inverse(base_quat, d.qvel[:3])
    base_ang_vel = quat_rotate_inverse(base_quat, d.qvel[3:6])

    phi_l = 2.0 * math.pi * phase
    phi_r = 2.0 * math.pi * ((phase + 0.5) % 1.0)
    gait_phase = jnp.array([jnp.sin(phi_l), jnp.cos(phi_l), jnp.sin(phi_r), jnp.cos(phi_r)])

    obs = jnp.concatenate([
        proj_grav,
        base_lin_vel,
        base_ang_vel,
        d.qpos[7:7+nu] - default_pose,
        d.qvel[6:6+nu],
        prev_act,
        cmd_vel,
        gait_phase
    ])
    return jnp.clip(obs, -20.0, 20.0)

# ================================================================
# 4. ENVIRONMENT STEP & RESET
# ================================================================
def env_reset(rng):
    rng_q, rng_v, rng_j, rng_mode, rng_cmd, rng_phase = jax.random.split(rng, 6)
    noise = jax.random.uniform(rng_j, (nq-7,), minval=-0.02, maxval=0.02)
    qpos  = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.005, maxval=0.005),
        default_qpos[7:] + noise
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.02, maxval=0.02)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)

    # 35% probability of zero velocity (Pure Stand-Still Balance training)
    # 65% probability of walking velocity command
    is_stand_mode = jax.random.bernoulli(rng_mode, p=0.35)
    walk_cmd = jax.random.uniform(
        rng_cmd, (3,),
        minval=jnp.array([0.10, -0.15, -0.30]),
        maxval=jnp.array([0.50,  0.15,  0.30]),
    )
    cmd_vel = jnp.where(is_stand_mode, jnp.zeros(3), walk_cmd)

    phase = jax.random.uniform(rng_phase, (), minval=0.0, maxval=1.0)
    return {
        "d": d, "prev_act": jnp.zeros(nu), "step": jnp.zeros((), jnp.int32),
        "phase": phase, "cmd_vel": cmd_vel,
    }

def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos = d.qpos
    qvel = d.qvel
    base_quat = qpos[3:7]

    proj_grav = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))
    base_lin_vel = quat_rotate_inverse(base_quat, qvel[:3])
    base_ang_vel = quat_rotate_inverse(base_quat, qvel[3:6])

    cmd_norm = jnp.linalg.norm(cmd_vel[:2])
    is_standing = jnp.where(cmd_norm < 0.05, 1.0, 0.0)
    is_walking  = 1.0 - is_standing

    # 1. Stand-Still Reward (Natural Balance when cmd_vel == 0)
    joint_dev = jnp.sum(jnp.square(qpos[7:7+nu] - default_pose))
    r_stand_pose = jnp.exp(-joint_dev / 0.10) * 3.0 * is_standing
    r_stand_vel  = jnp.exp(-jnp.sum(jnp.square(base_lin_vel[:2])) / 0.05) * 2.0 * is_standing

    # 2. Velocity Tracking in Robot Base Frame (Locomotion)
    vx_err = base_lin_vel[0] - cmd_vel[0]
    vy_err = base_lin_vel[1] - cmd_vel[1]
    # Asymmetric: linear brake penalty if exceeding commanded speed, exp reward if matching
    r_track_vx = jnp.where(
        vx_err > 0.1,
        jnp.exp(-jnp.square(vx_err) / 0.09) * 1.5,
        jnp.exp(-jnp.square(vx_err) / 0.15) * 2.5
    ) * is_walking
    r_track_vy  = jnp.exp(-jnp.square(vy_err) / 0.15) * 1.0 * is_walking
    r_track_yaw = jnp.exp(-jnp.square(base_ang_vel[2] - cmd_vel[2]) / 0.15) * 1.0

    # 3. Posture & Height (Active in all modes)
    r_upright = jnp.exp(-jnp.sum(jnp.square(proj_grav[:2])) / 0.04) * 2.0
    r_height  = jnp.exp(-jnp.square(qpos[2] - Z_NOMINAL) / 0.02) * 2.0
    r_alive   = 1.0

    # 4. Alternating Bipedal Gait Coordination (Active only when walking)
    l_z = d.site_xpos[L_FOOT_SITE_ID, 2]
    r_z = d.site_xpos[R_FOOT_SITE_ID, 2]
    l_c = jnp.where(l_z < CONTACT_Z_THR, 1.0, 0.0)
    r_c = jnp.where(r_z < CONTACT_Z_THR, 1.0, 0.0)
    l_stance = jnp.where(phase < STANCE_DUTY, 1.0, 0.0)
    r_stance = jnp.where(((phase + 0.5) % 1.0) < STANCE_DUTY, 1.0, 0.0)
    r_gait = (jnp.where(l_stance > 0.5, l_c, 1.0 - l_c) + jnp.where(r_stance > 0.5, r_c, 1.0 - r_c)) * 1.5 * is_walking

    # 5. Feet Air Time / Swing Clearance (Prevents foot dragging)
    swing_clearance = (
        jnp.where((1.0 - l_stance) > 0.5, jnp.clip(l_z - 0.03, 0.0, 0.1), 0.0) +
        jnp.where((1.0 - r_stance) > 0.5, jnp.clip(r_z - 0.03, 0.0, 0.1), 0.0)
    ) * 1.0 * is_walking

    # 6. Smoothness & Energy Penalties
    p_action_rate = jnp.sum(jnp.square(action - prev_action)) * 0.05
    p_torque      = jnp.sum(jnp.square(action)) * 0.01
    p_pitch_roll  = (jnp.square(base_ang_vel[0]) + jnp.square(base_ang_vel[1])) * 0.05

    total = (
        r_stand_pose + r_stand_vel +
        r_track_vx + r_track_vy + r_track_yaw +
        r_upright + r_height + r_alive +
        r_gait + swing_clearance -
        p_action_rate - p_torque - p_pitch_roll
    )
    return total * CTRL_DT

def env_step(state, action_and_rng):
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
    rew = compute_reward(d, env_act, prev_act, cmd_vel, phase)
    obs_out = get_obs(d, env_act, cmd_vel, new_phase)

    # Termination: pelvic height or tilt angle
    base_quat = d.qpos[3:7]
    proj_grav = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))
    terminated = jnp.logical_or(proj_grav[2] > TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new   = step + 1
    truncated  = step_new >= EPISODE_LEN
    done       = jnp.logical_or(terminated, truncated)

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
# 5. NETWORK INITIALIZATION
# ================================================================
network = ActorCritic(action_dim=nu)
rng     = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params  = network.init(ri, jnp.zeros((1, OBS_DIM_S2)))

# ================================================================
# 6. PPO HYPERPARAMETERS (State-of-the-Art Locomotion)
# ================================================================
NUM_ENVS     = 4096
ROLLOUT      = 24
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
N_EPOCHS     = 4
MINIBATCH    = 4096
TOTAL_STEPS  = 150_000_000
STEPS_PER_IT = NUM_ENVS * ROLLOUT   # 4096 x 24 = 98,304
N_ITERS      = TOTAL_STEPS // STEPS_PER_IT

print(f"[v9 CONFIG] Envs={NUM_ENVS} | Rollout={ROLLOUT} | Steps/iter={STEPS_PER_IT:,} | Total iters={N_ITERS}")
print(f"[v9 CONFIG] Obs={OBS_DIM_S2} | ActionScale={ACTION_SCALE} | Total steps={TOTAL_STEPS:,}")

lr_schedule = optax.cosine_decay_schedule(1e-3, N_ITERS, alpha=0.1)
tx          = optax.chain(optax.clip_by_global_norm(MAX_GRAD),
                          optax.adam(lr_schedule, eps=1e-5))
opt_state   = tx.init(params)

start_it = 1
cur = 0

if args.resume and os.path.exists(args.resume):
    print(f"[RESUME] Loading checkpoint from {args.resume}...", flush=True)
    ck_data = np.load(args.resume)
    flat_params = {k: jnp.array(ck_data[k]) for k in ck_data.files if k not in ["_step", "_it"]}
    params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    if "_step" in ck_data:
        cur = int(ck_data["_step"])
    if "_it" in ck_data:
        start_it = int(ck_data["_it"])
    print(f"[RESUME] Restored step={cur:,}, it={start_it}", flush=True)

rng_envs = jax.random.split(rng, NUM_ENVS)
states   = jax.vmap(env_reset)(rng_envs)

# ================================================================
# 7. ROLLOUT & PPO TRAINING LOGIC
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
        raw_act = mu + std * jax.random.normal(ra, mu.shape)
        lp   = jnp.clip(-0.5 * jnp.sum(
            jnp.square((raw_act - mu) / (std + 1e-8)) +
            2.0 * ls + math.log(2.0 * math.pi), axis=-1), -10., 10.)
        env_act = jnp.clip(raw_act, -1., 1.)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (env_act, r_resets))
        return (nst, p, r), (obs, raw_act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(_step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj

    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"], s["cmd_vel"], s["phase"]))(fst)
    _, _, nv_last = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        done  = jnp.logical_or(terms[t], truncs[t])
        delta = rews[t] + GAMMA * nxv * (1. - terms[t].astype(jnp.float32)) - vals[t]
        gae   = delta + GAMMA * LAM * (1. - done.astype(jnp.float32)) * gae
        return (gae, vals[t]), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last), jnp.arange(ROLLOUT - 1, -1, -1))
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
# 8. MAIN TRAINING LOOP
# ================================================================
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Starting training loop at {time.strftime('%Y-%m-%d %H:%M:%S')}...", flush=True)

t0 = time.time()
cur_ent = ENT_COEF

for it in range(start_it, N_ITERS + 1):
    cur += STEPS_PER_IT

    # Entropy coefficient decay over first 50M steps
    progress_50m = min(1.0, cur / 50_000_000)
    cur_ent = float(ENT_COEF * (1.0 - 0.9 * progress_50m))

    t_it = time.time()
    states, rng, fo, fa, flp, fadv, fret, ovf, mean_rew = collect_rollout(params, states, rng)

    n_samples = STEPS_PER_IT
    n_batches = n_samples // MINIBATCH
    perm_rng, rng = jax.random.split(rng)

    tot_loss, tot_kl = 0.0, 0.0
    for _ in range(N_EPOCHS):
        perm = jax.random.permutation(perm_rng, n_samples)
        for b in range(n_batches):
            idx = perm[b * MINIBATCH : (b + 1) * MINIBATCH]
            params, opt_state, loss, kl = ppo_minibatch_update(
                params, opt_state,
                fo[idx], fa[idx], flp[idx], fadv[idx], fret[idx], ovf[idx],
                cur_ent
            )
            tot_loss += float(loss)
            tot_kl   += float(kl)

    sps = STEPS_PER_IT / max(time.time() - t_it, 1e-4)
    ls_mean = float(jnp.mean(network.apply(params, fo[:1])[1]))

    status = "*** BALANCED WALKING ***" if mean_rew > 0.08 else ("stepping" if mean_rew > 0.04 else "...")
    print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | rew={mean_rew:.5f} | "
          f"loss={tot_loss/(N_EPOCHS*n_batches):.4f} | sps={sps:,.0f} | "
          f"ent={cur_ent:.4f} | log_std={ls_mean:.3f} | "
          f"t={time.time()-t0:.0f}s {status}", flush=True)

    # Save checkpoint every 100 iters (~10M steps) or final
    if it % 100 == 0 or it == N_ITERS:
        ck = f"{CKPT_DIR}/apollo_stage2_v9_step_{cur}.npz"
        flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
        flat_np["_step"] = np.array(cur)
        flat_np["_it"]   = np.array(it)
        np.savez(ck, **flat_np)

        latest_ck = f"{CKPT_DIR}/apollo_stage2_v9_latest.npz"
        import shutil as _shutil
        _shutil.copy(ck, latest_ck)
        print(f"  -> saved latest checkpoint: {latest_ck}", flush=True)

        # Push checkpoint to GitHub via REST API
        gh_token = ""
        token_file = "/content/github_token.txt"
        if os.path.exists(token_file):
            try:
                with open(token_file, "r", encoding="utf-8") as tf:
                    gh_token = tf.read().strip()
            except Exception:
                pass
        if not gh_token:
            gh_token = os.environ.get("GITHUB_TOKEN", "")

        if gh_token:
            try:
                import base64, urllib.request, json as _json
                repo = "tranvanmanh9325/medical-science"
                api_path = "colab_output/checkpoints_stage2/apollo_stage2_v9_latest.npz"
                api_url = f"https://api.github.com/repos/{repo}/contents/{api_path}"
                with open(latest_ck, "rb") as f:
                    ck_b64 = base64.b64encode(f.read()).decode()
                sha = None
                try:
                    req_get = urllib.request.Request(api_url, headers={"Authorization": f"token {gh_token}", "User-Agent": "ColabTrainer"})
                    with urllib.request.urlopen(req_get, timeout=15) as r:
                        sha = _json.loads(r.read())["sha"]
                except Exception:
                    pass
                payload = {"message": f"[skip ci] sync v9 checkpoint step={cur} it={it}", "content": ck_b64, "branch": "main"}
                if sha:
                    payload["sha"] = sha
                req_put = urllib.request.Request(
                    api_url,
                    data=_json.dumps(payload).encode(),
                    headers={"Authorization": f"token {gh_token}", "Content-Type": "application/json", "User-Agent": "ColabTrainer"},
                    method="PUT"
                )
                with urllib.request.urlopen(req_put, timeout=120) as r:
                    r.read()
                print(f"  -> GitHub REST API push OK (step={cur:,})", flush=True)
            except Exception as e:
                print(f"  [WARN] GitHub push failed: {e}", flush=True)

# Final checkpoint save
flat_np = {k: np.array(v) for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()}
flat_np["_step"] = np.array(cur)
flat_np["_it"]   = np.array(it)
np.savez(f"{CKPT_DIR}/apollo_stage2_v9_final.npz", **flat_np)
print("\nSTAGE 2 v9 TRAINING COMPLETE! (Natural Stand & Locomotion)", flush=True)
print(f"Total steps: {cur:,} | Final Checkpoint: {CKPT_DIR}/apollo_stage2_v9_final.npz", flush=True)
