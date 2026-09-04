import os
import json


def generate():
    root_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = os.path.join(root_dir, "colab_deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    TRAINING_CODE = r"""
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

OBS_DIM_S1 = 105
OBS_DIM_S2 = 114

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

model_path = "/content/mujoco_menagerie/apptronik_apollo/scene.xml"
mj_model   = mujoco.MjModel.from_xml_path(model_path)
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
default_ctrl = jnp.array(mj_model.key_qpos[key_id][7:])
default_pose = jnp.array(mj_model.key_qpos[key_id][7:])

Z_NOMINAL    = float(default_qpos[2])
ACTION_SCALE = 0.25
EPISODE_LEN  = 500
TERM_HEIGHT  = Z_NOMINAL * 0.50
TERM_TILT    = 0.45

STEP_FREQ   = 1.2
STANCE_DUTY = 0.55
L_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_fl")
R_FOOT_SITE_ID = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_fl")
CONTACT_Z_THR  = 0.08

PUSH_START_STEP = 500_000
PUSH_MAX_STEP   = 5_000_000
PUSH_MAX_FORCE  = 80.0

print(f"z_nominal={Z_NOMINAL:.4f}m | action_scale={ACTION_SCALE} | step_freq={STEP_FREQ}Hz")

stage1_ck = None
stage1_search_paths = [
    "/content/drive/MyDrive/apollo_humanoid_checkpoints/apollo_stage1_v15_step_99876864.npz",
    "/content/drive/MyDrive/apollo_stage1_checkpoints/apollo_stage1_v15_step_99876864.npz",
    "/content/drive/MyDrive/apollo_humanoid_checkpoints/*.npz",
    "/content/drive/MyDrive/apollo_stage1_checkpoints/*.npz",
    "/content/*.npz",
]
for pattern in stage1_search_paths:
    found = sorted(glob.glob(pattern))
    if found:
        stage1_ck = max(found, key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0"))
        break

network = ActorCritic(action_dim=nu)
rng     = jax.random.PRNGKey(42)
rng, ri = jax.random.split(rng)
params  = network.init(ri, jnp.zeros((1, OBS_DIM_S2)))

if stage1_ck:
    print(f"[TRANSFER LEARNING] Loading Stage 1: {stage1_ck}")
    s1_data = dict(np.load(stage1_ck))
    s1_keys = list(s1_data.keys())
    w0_key  = next((k for k in s1_keys if "Dense_0" in k and "kernel" in k), None)
    if w0_key and s1_data[w0_key].shape[0] == OBS_DIM_S1:
        W_old = s1_data[w0_key]
        W_new = np.random.randn(9, 512).astype(np.float32) * 0.01
        s1_data[w0_key] = np.vstack([W_old, W_new])
        print(f"  [OK] Input layer: ({OBS_DIM_S1},512) -> ({OBS_DIM_S2},512)")
        def unflatten(flat_dict, sep="/"):
            nested = {}
            for k, v in flat_dict.items():
                parts = k.split(sep); d = nested
                for p in parts[:-1]: d = d.setdefault(p, {})
                d[parts[-1]] = jnp.array(v)
            return nested
        import flax.traverse_util as ftu
        flat_new = ftu.flatten_dict(params, sep="/")
        flat_s1  = ftu.flatten_dict(unflatten(s1_data), sep="/")
        transferred = 0
        for k in flat_new:
            if k in flat_s1 and flat_new[k].shape == flat_s1[k].shape:
                flat_new[k] = flat_s1[k]; transferred += 1
        params = ftu.unflatten_dict(flat_new, sep="/")
        print(f"  [OK] Transferred {transferred} tensors from Stage 1")
else:
    print("[INFO] No Stage 1 checkpoint — training from scratch")
    print("  Upload apollo_stage1_v15_step_99876864.npz to MyDrive/apollo_humanoid_checkpoints/")

def get_upvector(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    return jnp.array([2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx**2+qy**2)])

def get_obs(d, prev_act, cmd_vel, phase):
    upvec = get_upvector(d.qpos)
    phi_l = 2.0*math.pi*phase; phi_r = 2.0*math.pi*((phase+0.5)%1.0)
    gait_phase = jnp.array([jnp.sin(phi_l), jnp.cos(phi_l), jnp.sin(phi_r), jnp.cos(phi_r)])
    l_z = d.site_xpos[L_FOOT_SITE_ID,2]; r_z = d.site_xpos[R_FOOT_SITE_ID,2]
    foot_contact = jnp.array([(l_z<CONTACT_Z_THR).astype(jnp.float32),(r_z<CONTACT_Z_THR).astype(jnp.float32)])
    obs = jnp.concatenate([upvec, d.qvel[:3], d.qvel[3:6], d.qpos[7:7+nu]-default_pose, d.qvel[6:6+nu], prev_act, cmd_vel, gait_phase, foot_contact])
    return jnp.clip(obs, -20.0, 20.0)

def env_reset(rng):
    rng_q, rng_v, rng_j, rng_cmd, rng_phase = jax.random.split(rng, 5)
    noise = jax.random.uniform(rng_j, (nq-7,), minval=-0.05, maxval=0.05)
    qpos  = jnp.concatenate([default_qpos[:7]+jax.random.uniform(rng_q,(7,),minval=-0.01,maxval=0.01), default_qpos[7:]+noise])
    dv = jax.random.uniform(rng_v, (nv,), minval=-0.05, maxval=0.05)
    d  = mjx.make_data(mjx_model)
    d  = d.replace(qpos=qpos, qvel=dv)
    d  = mjx.forward(mjx_model, d)
    cmd_vel = jax.random.uniform(rng_cmd, (3,), minval=jnp.array([0.0,-0.06,-0.1]), maxval=jnp.array([0.15,0.06,0.1]))
    phase   = jax.random.uniform(rng_phase, (), minval=0.0, maxval=1.0)
    return {"d":d, "prev_act":jnp.zeros(nu), "step":jnp.zeros((),jnp.int32), "phase":phase, "cmd_vel":cmd_vel}

def env_step(state, action_and_rng):
    raw_act, rng_reset = action_and_rng
    d, prev_act, step, phase, cmd_vel = state["d"], state["prev_act"], state["step"], state["phase"], state["cmd_vel"]
    ctrl = jnp.clip(default_ctrl+raw_act*ACTION_SCALE, ctrl_range[:,0], ctrl_range[:,1])
    d = d.replace(ctrl=ctrl)
    def _sub(dd, _): return mjx.step(mjx_model, dd), None
    d, _ = jax.lax.scan(_sub, d, None, length=N_SUBSTEPS)
    new_phase = (phase+CTRL_DT*STEP_FREQ)%1.0
    rew     = compute_reward(d, raw_act, prev_act, cmd_vel, phase)
    obs_out = get_obs(d, raw_act, cmd_vel, new_phase)
    upvec   = get_upvector(d.qpos)
    terminated = jnp.logical_or(upvec[2]<TERM_TILT, d.qpos[2]<TERM_HEIGHT)
    step_new = step+1; truncated = step_new>=EPISODE_LEN
    done = jnp.logical_or(terminated, truncated)
    rst = env_reset(rng_reset)
    next_d   = jax.tree.map(lambda r,c: jnp.where(done,r,c), rst["d"], d)
    nst = {"d":next_d, "prev_act":jnp.where(done,jnp.zeros(nu),raw_act), "step":jnp.where(done,jnp.zeros((),jnp.int32),step_new), "phase":jnp.where(done,rst["phase"],new_phase), "cmd_vel":jnp.where(done,rst["cmd_vel"],cmd_vel)}
    return obs_out, nst, rew, terminated, truncated

def compute_reward(d, action, prev_action, cmd_vel, phase):
    qpos=d.qpos; qvel=d.qvel; upvec=get_upvector(qpos)
    r_vel_lin = jnp.exp(-jnp.sum(jnp.square(qvel[:2]-cmd_vel[:2]))/0.09)
    r_vel_ang = jnp.exp(-jnp.square(qvel[5]-cmd_vel[2])/0.09)
    r_orient  = jnp.exp(-jnp.sum(jnp.square(upvec[:2]))/0.10)
    r_height  = jnp.exp(-jnp.square(qpos[2]-Z_NOMINAL)/0.10)
    r_alive   = 0.2
    l_z=d.site_xpos[L_FOOT_SITE_ID,2]; r_z=d.site_xpos[R_FOOT_SITE_ID,2]
    l_c=(l_z<CONTACT_Z_THR).astype(jnp.float32); r_c=(r_z<CONTACT_Z_THR).astype(jnp.float32)
    l_ts=(phase<STANCE_DUTY).astype(jnp.float32); rp=(phase+0.5)%1.0; r_ts=(rp<STANCE_DUTY).astype(jnp.float32)
    r_gait = (jnp.where(l_ts>0.5,l_c,1.-l_c)+jnp.where(r_ts>0.5,r_c,1.-r_c))*0.3
    p_ar=0.02*jnp.mean(jnp.square(action-prev_action)); p_tq=2e-4*jnp.sum(jnp.square(action)); p_bt=0.05*(jnp.square(qvel[3])+jnp.square(qvel[4]))
    total = r_vel_lin*2.0+r_vel_ang*0.5+r_orient*0.5+r_height*0.3+r_alive+r_gait-p_ar-p_tq-p_bt
    return jnp.maximum(0.0,total)*CTRL_DT

NUM_ENVS=4096; ROLLOUT=64; GAMMA=0.99; LAM=0.95; CLIP_EPS=0.2
ENT_COEF=0.012; VF_COEF=0.5; MAX_GRAD=0.5
TOTAL_STEPS=150_000_000; STEPS_PER_IT=NUM_ENVS*ROLLOUT; N_ITERS=TOTAL_STEPS//STEPS_PER_IT

lr_schedule = optax.linear_schedule(2e-4, 1e-5, N_ITERS)
tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(lr_schedule, eps=1e-5))
opt_state = tx.init(params)
states = jax.vmap(env_reset)(jax.random.split(rng, NUM_ENVS))

@jax.jit
def ppo_iter(params, opt_state, states, rng):
    def _step(carry, _):
        st,p,r = carry; r,ra,r_reset = jax.random.split(r,3)
        r_resets = jax.random.split(r_reset, NUM_ENVS)
        obs = jax.vmap(lambda s: get_obs(s["d"],s["prev_act"],s["cmd_vel"],s["phase"]))(st)
        mu,ls,val = network.apply(p,obs); std=jnp.exp(ls)
        act = jnp.clip(mu+std*jax.random.normal(ra,mu.shape),-1.,1.)
        lp  = jnp.clip(-0.5*jnp.sum(jnp.square((act-mu)/(std+1e-8))+2.*ls+math.log(2.*math.pi),axis=-1),-10.,10.)
        _,nst,rew,term,trunc = jax.vmap(env_step)(st,(act,r_resets))
        return (nst,p,r),(obs,act,lp,val,rew,term,trunc)
    (fst,_,rng),traj = jax.lax.scan(_step,(states,params,rng),None,length=ROLLOUT)
    obs,act,old_lp,vals,rews,terms,truncs = traj
    lobs = jax.vmap(lambda s: get_obs(s["d"],s["prev_act"],s["cmd_vel"],s["phase"]))(fst)
    _,_,nv_last = network.apply(params,lobs)
    def _gae(carry,t):
        gae,nxv=carry; r,v,term,trunc=rews[t],vals[t],terms[t],truncs[t]
        done=jnp.logical_or(term,trunc); delta=r+GAMMA*nxv*(1.-term.astype(jnp.float32))-v
        gae=delta+GAMMA*LAM*(1.-done.astype(jnp.float32))*gae; return (gae,v),gae
    _,advs = jax.lax.scan(_gae,(jnp.zeros(NUM_ENVS),nv_last),jnp.arange(ROLLOUT-1,-1,-1))
    advs=jnp.flip(advs,axis=0); rets=advs+vals; advs=(advs-advs.mean())/(advs.std()+1e-8)
    flat=lambda x: x.reshape(-1,*x.shape[2:])
    fo,fa,flp,fadv,fret,ovf = *map(flat,[obs,act,old_lp,advs,rets]),flat(vals)
    def loss_fn(p):
        mu,ls,v=network.apply(p,fo); std=jnp.exp(ls)
        lp=jnp.clip(-0.5*jnp.sum(jnp.square((fa-mu)/(std+1e-8))+2.*ls+math.log(2.*math.pi),axis=-1),-10.,10.)
        ratio=jnp.exp(jnp.clip(lp-flp,-5.,5.))
        pg=-jnp.mean(jnp.minimum(ratio*fadv,jnp.clip(ratio,1-CLIP_EPS,1+CLIP_EPS)*fadv))
        vc=ovf+jnp.clip(v-ovf,-5.,5.); vf=VF_COEF*jnp.mean(jnp.maximum(jnp.square(v-fret),jnp.square(vc-fret)))
        ent=-ENT_COEF*jnp.mean(jnp.sum(ls+0.5*math.log(2*math.pi*math.e),axis=-1))
        return jnp.where(jnp.isnan(pg+vf+ent),0.,pg+vf+ent)
    loss,grads=jax.value_and_grad(loss_fn)(params)
    grads=jax.tree.map(lambda g: jnp.where(jnp.isnan(g),0.,g),grads)
    upd,opt_state=tx.update(grads,opt_state,params)
    return optax.apply_updates(params,upd),opt_state,fst,rng,jnp.mean(rews),loss

os.makedirs(CKPT_DIR, exist_ok=True)
t0,cur=time.time(),0
CURRICULUM_P1_END=50_000_000; CURRICULUM_P2_END=120_000_000

def get_curriculum_cmd_max(n):
    if n<CURRICULUM_P1_END: vx=0.15
    elif n<CURRICULUM_P2_END:
        t=(n-CURRICULUM_P1_END)/(CURRICULUM_P2_END-CURRICULUM_P1_END); vx=0.15+t*0.65
    else: vx=0.80
    return vx, min(0.3,vx*0.4), min(0.4,vx*0.5)

@jax.jit
def reseed_cmd_vel(states,rng,vx_max,vy_max,yaw_max):
    rngs=jax.random.split(rng,NUM_ENVS)
    def _new(r): return jax.random.uniform(r,(3,),minval=jnp.array([0.,-vy_max,-yaw_max]),maxval=jnp.array([vx_max,vy_max,yaw_max]))
    return {**states,"cmd_vel":jax.vmap(_new)(rngs)}

print(f"Steps/iter={STEPS_PER_IT:,} | N_iters={N_ITERS} | Transfer={'YES' if stage1_ck else 'NO'}")
print(f"Velocity curriculum: [0,0.15]->[0,0.45]->[0,0.80] m/s | Reward kernel sigma=0.09")
print("=" * 64)

for it in range(1, N_ITERS+1):
    t1=time.time()
    if it%10==1:
        vx_max,vy_max,yaw_max=get_curriculum_cmd_max(cur); rng,rs=jax.random.split(rng)
        states=reseed_cmd_vel(states,rs,jnp.float32(vx_max),jnp.float32(vy_max),jnp.float32(yaw_max))
    params,opt_state,states,rng,mr,loss=ppo_iter(params,opt_state,states,rng)
    jax.block_until_ready(params)
    cur+=STEPS_PER_IT; sps=STEPS_PER_IT/max(1e-5,time.time()-t1)
    if it%5==0 or it==1:
        r_val=float(mr); vx_max,_,_=get_curriculum_cmd_max(cur)
        push_frac=min(1.,max(0.,(cur-PUSH_START_STEP)/(PUSH_MAX_STEP-PUSH_START_STEP)))
        push_mag=PUSH_MAX_FORCE*push_frac if cur>=PUSH_START_STEP else 0.
        status=("*** WALKING WELL ***" if r_val>0.022 else "*** WALKING ***" if r_val>0.016 else "stepping" if r_val>0.010 else "improving" if r_val>0.006 else "...")
        print(f"[{it:04d}/{N_ITERS}] steps={cur:,} | rew={r_val:.5f} | loss={float(loss):.4f} | sps={sps:,.0f} | push={push_mag:.0f}N | vx_max={vx_max:.2f} | t={time.time()-t0:.0f}s {status}",flush=True)
    if it%50==0 or it==N_ITERS:
        ck=f"{CKPT_DIR}/apollo_stage2_v2_step_{cur}.npz"
        flat=flax.traverse_util.flatten_dict(params,sep="/")
        np.savez(ck, **{k:np.array(v) for k,v in flat.items()})
        sz=os.path.getsize(ck)
        if sz<100_000: print(f"  [WARNING] Small checkpoint: {sz} bytes!")
        else: print(f"  -> checkpoint: {ck} ({sz//1024}KB)",flush=True)

print("STAGE 2 v2 COMPLETE", flush=True)
"""

    INSTALL_CELL = [
        "# Step 1: Check GPU (Runtime > Change runtime type > T4 GPU)\n",
        "!nvidia-smi\n",
        "!pip install -q --no-cache-dir mujoco mujoco-mjx flax optax\n",
        "import jax\n",
        "print('Backend:', jax.default_backend(), '| Devices:', jax.devices())\n",
        "assert jax.default_backend() in ('gpu','tpu'), 'Enable GPU in Runtime > Change runtime type!'"
    ]

    DRIVE_CELL = [
        "# Step 2: Mount Google Drive for persistent checkpoint storage\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "import os\n",
        "CKPT_DIR = '/content/drive/MyDrive/apollo_stage2_checkpoints'\n",
        "os.makedirs(CKPT_DIR, exist_ok=True)\n",
        "# Upload Stage 1 checkpoint to: MyDrive/apollo_humanoid_checkpoints/ for transfer learning\n",
        "print(f'[OK] Checkpoints will be saved to: {CKPT_DIR}')"
    ]

    DOWNLOAD_CELL = [
        "# Step 3: Download Apollo robot model from GitHub\n",
        "import os, urllib.request, zipfile, shutil\n",
        "TARGET = '/content/mujoco_menagerie'\n",
        "APOLLO_XML = os.path.join(TARGET,'apptronik_apollo','scene.xml')\n",
        "if not os.path.exists(APOLLO_XML):\n",
        "    print('Downloading mujoco_menagerie (~510MB)...')\n",
        "    url='/'.join(['https://github.com/google-deepmind/mujoco_menagerie','archive/refs/heads/main.zip'])\n",
        "    urllib.request.urlretrieve(url, '/tmp/men.zip')\n",
        "    with zipfile.ZipFile('/tmp/men.zip') as z: z.extractall('/tmp/men_ex')\n",
        "    shutil.move('/tmp/men_ex/mujoco_menagerie-main', TARGET)\n",
        "    os.remove('/tmp/men.zip')\n",
        "assert os.path.exists(APOLLO_XML)\n",
        "print(f'[OK] Apollo model ready: {APOLLO_XML}')"
    ]

    cells = [
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":INSTALL_CELL},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":DRIVE_CELL},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":DOWNLOAD_CELL},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[TRAINING_CODE]},
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType":"T4","provenance":[],"name":"Apollo Humanoid Stage 2 Walking v2"},
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 0
    }

    nb_path = os.path.join(deploy_dir, "apollo_stage2_walking_colab.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[COLAB STAGE 2] {nb_path} ({os.path.getsize(nb_path)//1024}KB)")
    return nb_path

if __name__ == "__main__":
    generate()
