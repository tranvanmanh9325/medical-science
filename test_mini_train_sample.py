import os
import sys
import time
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import mujoco
from mujoco import mjx

print("=" * 64)
print("  [DEMO MẪU NHỎ] CHẠY THỬ QUY TRÌNH HUẤN LUYỆN APOLLO PPO (MJX)")
print("=" * 64)
print("JAX Backend:", jax.default_backend())
print("Thiết bị:", jax.devices())

# 1. Mạng nơ-ron Actor-Critic
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in (256, 128):
            x = nn.elu(nn.Dense(h)(x))
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        log_std = jnp.clip(log_std, -3.0, 0.5)
        value = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value

# 2. Khởi tạo mô hình MuJoCo MJX
model_path = "google_deepmind_menagerie/apptronik_apollo/scene.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
SIM_DT, N_SUBSTEPS = 0.002, 5
CTRL_DT = SIM_DT * N_SUBSTEPS

mj_model.opt.timestep = SIM_DT
mj_model.opt.iterations = 4
mj_model.opt.ls_iterations = 4
mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

mjx_model = mjx.put_model(mj_model)
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
ctrl_range = jnp.array(mj_model.actuator_ctrlrange)

key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id < 0:
    key_id = 0
default_qpos = jnp.array(mj_model.key_qpos[key_id])
default_ctrl = jnp.array(mj_model.key_ctrl[key_id])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL = float(default_qpos[2])
TRACKING_SIGMA = 0.25
ACTION_SCALE = 0.3
EPISODE_LEN = 500
TERM_HEIGHT = Z_NOMINAL * 0.75
TERM_TILT = 0.5
OBS_DIM = 3 + 3 + 3 + nu + nu + nu

def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx ** 2 + qy ** 2),
    ])

def get_obs(d, prev_act):
    upvec = get_upvector(d.qpos)
    linvel = d.qvel[:3]
    angvel = d.qvel[3:6]
    jpos = d.qpos[7:7 + nu] - default_pose
    jvel = d.qvel[6:6 + nu]
    obs = jnp.concatenate([upvec, linvel, angvel, jpos, jvel, prev_act])
    return jnp.clip(obs, -20.0, 20.0)

def compute_reward(qpos, qvel, torques, action, prev_action):
    upvec = get_upvector(qpos)
    tracking_lin_vel = jnp.exp(-jnp.sum(jnp.square(qvel[:2])) / TRACKING_SIGMA)
    tracking_ang_vel = jnp.exp(-jnp.square(qvel[5]) / TRACKING_SIGMA)
    cost_linvel_z = jnp.square(qvel[2])
    cost_angvel_xy = jnp.sum(jnp.square(qvel[3:5]))
    cost_orientation = jnp.sum(jnp.square(upvec[:2]))
    jpos = qpos[7:7 + nu]
    cost_stand_still = jnp.sum(jnp.abs(jpos - default_pose))
    cost_torques = (jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques)))
    cost_action_rate = jnp.sum(jnp.square(action - prev_action))
    out_lo = -jnp.clip(jpos - ctrl_range[:nu, 0], -jnp.inf, 0.0)
    out_hi = jnp.clip(jpos - ctrl_range[:nu, 1], 0.0, jnp.inf)
    cost_dof_limits = jnp.sum(out_lo + out_hi)
    total = (
        1.0 * tracking_lin_vel
        + 0.5 * tracking_ang_vel
        - 2.0 * cost_linvel_z
        - 0.05 * cost_angvel_xy
        - 1.0 * cost_orientation
        - 0.5 * cost_stand_still
        - 1e-4 * cost_torques
        - 0.01 * cost_action_rate
        - 10.0 * cost_dof_limits
    )
    return total * CTRL_DT

def env_reset(rng):
    rng_q, rng_v, rng_j = jax.random.split(rng, 3)
    noise = jax.random.uniform(rng_j, (nq - 7,), minval=0.90, maxval=1.10)
    qpos = jnp.concatenate([
        default_qpos[:7] + jax.random.uniform(rng_q, (7,), minval=-0.01, maxval=0.01),
        default_qpos[7:] * noise
    ])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.02, maxval=0.02)
    d = mjx.make_data(mjx_model)
    d = d.replace(qpos=qpos, qvel=dv)
    d = mjx.forward(mjx_model, d)
    prev_act = jnp.zeros(nu)
    return {"d": d, "prev_act": prev_act, "step": jnp.zeros((), jnp.int32)}

def env_step(state, action_and_rng):
    raw_act, rng_reset = action_and_rng
    d, prev_act, step = state["d"], state["prev_act"], state["step"]
    ctrl = jnp.clip(default_ctrl + raw_act * ACTION_SCALE, ctrl_range[:, 0], ctrl_range[:, 1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)
    torques = d.qfrc_actuator[6:6 + nu]
    rew = compute_reward(d.qpos, d.qvel, torques, raw_act, prev_act)
    obs_out = get_obs(d, raw_act)
    upvec = get_upvector(d.qpos)
    terminated = jnp.logical_or(upvec[2] < TERM_TILT, d.qpos[2] < TERM_HEIGHT)
    step_new = step + 1
    truncated = step_new >= EPISODE_LEN
    done = jnp.logical_or(terminated, truncated)
    reset_state = env_reset(rng_reset)
    next_d = jax.tree.map(lambda r, c: jnp.where(done, r, c), reset_state["d"], d)
    next_act = jnp.where(done, jnp.zeros(nu), raw_act)
    next_step = jnp.where(done, jnp.zeros((), jnp.int32), step_new)
    nst = {"d": next_d, "prev_act": next_act, "step": next_step}
    return obs_out, nst, rew, terminated, truncated

# Cấu hình mẫu nhỏ (Chạy nhanh thử nghiệm)
NUM_ENVS = 64
ROLLOUT = 16
TOTAL_STEPS = NUM_ENVS * ROLLOUT * 5
N_ITERS = 5

print(f"\n[CẤU HÌNH TEST NHANH]")
print(f"- Số môi trường song song: {NUM_ENVS} Envs")
print(f"- Số bước Rollout mỗi vòng: {ROLLOUT}")
print(f"- Tổng số vòng lặp test: {N_ITERS} iters ({TOTAL_STEPS:,} steps)")

network = ActorCritic(action_dim=nu)
rng = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params = network.init(ri, jnp.zeros((1, OBS_DIM)))
tx = optax.adam(3e-4)
opt_state = tx.init(params)

rng_envs = jax.random.split(rng, NUM_ENVS)
states = jax.vmap(env_reset)(rng_envs)

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st, p, r = carry
        r, ra, r_reset = jax.random.split(r, 3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(st)
        mu, ls, val = network.apply(p, obs)
        std = jnp.exp(ls)
        act = jnp.clip(mu + std * jax.random.normal(ra, mu.shape), -1.0, 1.0)
        lp = -0.5 * jnp.sum(jnp.square((act - mu) / (std + 1e-8)) + 2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        _, nst, rew, term, trunc = jax.vmap(env_step)(st, (act, r_resets))
        return (nst, p, r), (obs, act, lp, val, rew, term, trunc)

    (fst, _, rng), traj = jax.lax.scan(_step, (states, params, rng), None, length=ROLLOUT)
    obs, act, old_lp, vals, rews, terms, truncs = traj
    lobs = jax.vmap(lambda s: get_obs(s["d"], s["prev_act"]))(fst)
    _, _, nv_last = network.apply(params, lobs)

    def _gae(carry, t):
        gae, nxv = carry
        r, v, term, trunc = rews[t], vals[t], terms[t], truncs[t]
        done = jnp.logical_or(term, trunc)
        delta = r + 0.99 * nxv * (1.0 - term.astype(jnp.float32)) - v
        gae = delta + 0.99 * 0.95 * (1.0 - done.astype(jnp.float32)) * gae
        return (gae, v), gae

    _, advs = jax.lax.scan(_gae, (jnp.zeros(NUM_ENVS), nv_last), jnp.arange(ROLLOUT - 1, -1, -1))
    advs = jnp.flip(advs, axis=0)
    rets = advs + vals
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    fo, fa, flp, fadv, fret = map(flat, [obs, act, old_lp, advs, rets])

    def loss_fn(p):
        mu, ls, v = network.apply(p, fo)
        std = jnp.exp(ls)
        lp = -0.5 * jnp.sum(jnp.square((fa - mu) / (std + 1e-8)) + 2.0 * ls + math.log(2.0 * math.pi), axis=-1)
        ratio = jnp.exp(jnp.clip(lp - flp, -5.0, 5.0))
        pg = -jnp.mean(jnp.minimum(ratio * fadv, jnp.clip(ratio, 0.8, 1.2) * fadv))
        vf = 0.5 * jnp.mean(jnp.square(v - fret))
        ent = -0.01 * jnp.mean(jnp.sum(ls + 0.5 * math.log(2 * math.pi * math.e), axis=-1))
        return pg + vf + ent

    loss, grads = jax.value_and_grad(loss_fn)(params)
    upd, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, upd)
    return params, opt_state, fst, rng, jnp.mean(rews), loss

print("\n>>> BẮT ĐẦU CHẠY THỬ CÁC VÒNG LẶP HUẤN LUYỆN...")
t0 = time.time()
cur_steps = 0

for it in range(1, N_ITERS + 1):
    t_start = time.time()
    params, opt_state, states, rng, mean_rew, loss_val = ppo_iter(params, opt_state, states, rng)
    jax.block_until_ready(params)
    cur_steps += NUM_ENVS * ROLLOUT
    dt = max(1e-4, time.time() - t_start)
    sps = (NUM_ENVS * ROLLOUT) / dt

    r_val = float(mean_rew)
    status = "Đang thích nghi thăng bằng (Learning)"
    print(f"[{it:02d}/{N_ITERS}] Steps tích lũy: {cur_steps:,} | Phần thưởng (Reward): {r_val:+.5f} | Loss: {float(loss_val):.4f} | Tốc độ: {sps:,.0f} bước/s | Trạng thái: {status}")

# Lưu thử checkpoint
os.makedirs("test_checkpoints", exist_ok=True)
ck_path = f"test_checkpoints/apollo_mini_sample_step_{cur_steps}.npz"
flat_dict = flax.traverse_util.flatten_dict(params, sep="/")
jnp.savez(ck_path, **flat_dict)

print("\n" + "=" * 64)
print(f"  [KẾT QUẢ THÀNH CÔNG 100%]")
print(f"  - Đã hoàn thành huấn luyện mẫu nhỏ: {cur_steps:,} steps")
print(f"  - File Checkpoint mẫu đã được tạo tại: {ck_path}")
print(f"  - Thời gian chạy thử: {time.time() - t0:.2f}s")
print("=" * 64)
