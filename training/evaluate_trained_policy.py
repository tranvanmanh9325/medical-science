import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from training.env_apollo_mjx import ApolloMJXEnvV2
from training.ppo_mjx_trainer import ActorCritic

def evaluate_policy():
    print("==================================================================")
    print(" [TRAINED POLICY COMPREHENSIVE EVALUATION] Apptronik Apollo       ")
    print("==================================================================")

    # 1. Load Checkpoint
    ckpt_dir = os.path.join(root_dir, "kaggle_output", "checkpoints")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".npz")])
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in kaggle_output/checkpoints")
    
    latest_ckpt = ckpts[-1]
    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
    print(f"Loading Latest Trained Policy: {latest_ckpt}...")

    # Load weights
    npz_data = np.load(ckpt_path)
    print(f"Loaded {len(npz_data.files)} parameter tensors from checkpoint.")

    # 2. Reconstruct Flax Parameter Dictionary
    flat_params = {k: jnp.array(npz_data[k]) for k in npz_data.files}
    import flax.traverse_util
    params = flax.traverse_util.unflatten_dict(flat_params, sep='/')

    # 3. Initialize MuJoCo Physical Model
    xml_path = os.path.join(root_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    env = ApolloMJXEnvV2(xml_path)
    network = ActorCritic(action_dim=env.action_dim)

    # Forward Policy Function
    @jax.jit
    def policy_act(obs):
        actor_mean, _, _ = network.apply(params, obs[None, :])
        return actor_mean[0]

    # 4. Run 1,000 Step Physical Rollout (Autonomous Standing & Balance Test)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    action_hist = np.zeros((3, env.action_dim))
    cmd = np.array([0.5, 0.0, 0.0])  # Forward walking command
    phase = 0.0
    dt = mj_model.opt.timestep * 4

    heights = []
    tilts = []
    torques = []
    fallen = False

    print("\nRunning 1,000 Step Autonomous Evaluation Rollout...")
    for step in range(1000):
        # 1. Update Phase
        phase = (phase + 2.0 * np.pi * env.gait_frequency * dt) % (2.0 * np.pi)

        # 2. Construct Sensory Observation
        root_z = np.array([mj_data.qpos[2]])
        root_quat = mj_data.qpos[3:7]
        qw, qx, qy, qz = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
        proj_grav = np.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            -(qw * qw - qx * qx - qy * qy + qz * qz)
        ])
        root_linvel = mj_data.qvel[:3]
        root_angvel = mj_data.qvel[3:6]
        joint_pos = mj_data.qpos[7:7 + env.nu] - env.default_qpos[7:7 + env.nu]
        joint_vel = mj_data.qvel[6:6 + env.nu]
        phase_harmonics = np.array([np.sin(phase), np.cos(phase)])
        foot_forces = np.array([396.8, 396.8]) / 400.0
        flattened_hist = action_hist.flatten()

        obs = np.concatenate([
            root_z, root_quat, proj_grav, root_linvel, root_angvel,
            joint_pos, joint_vel, cmd, phase_harmonics, foot_forces,
            flattened_hist
        ])

        # 3. Query Policy Neural Network
        action = np.array(policy_act(jnp.array(obs)))
        action = np.clip(action, -1.0, 1.0)

        # 4. Apply to MuJoCo Actuators
        ctrl_targets = env.default_ctrl + (action * env.action_scale)
        mj_data.ctrl[:] = np.clip(ctrl_targets, env.ctrl_range[:, 0], env.ctrl_range[:, 1])

        # 5. Step Physics 4 times
        for _ in range(4):
            mujoco.mj_step(mj_model, mj_data)

        # 6. Record Metrics
        heights.append(mj_data.qpos[2])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_deg = np.arccos(np.clip(up_z, -1.0, 1.0)) * 180.0 / np.pi
        tilts.append(tilt_deg)
        torques.append(np.mean(np.abs(mj_data.qfrc_actuator)))

        # Update action history
        action_hist = np.vstack([action[None, :], action_hist[:2]])

        if mj_data.qpos[2] < 0.65:
            fallen = True
            print(f"Robot fell at step {step} (Height: {mj_data.qpos[2]:.2f}m)")
            break

    # Summary Metrics
    print("\n==================================================================")
    print(" [EVALUATION REPORT & QUANTITATIVE METRICS]                       ")
    print("==================================================================")
    print(f" - Rollout Completion Rate : {step + 1} / 1000 Steps ({'SUCCESS - NO FALL' if not fallen else 'FAILED'})")
    print(f" - Average Root Height Z   : {np.mean(heights):.3f} m (Nominal: 1.016 m)")
    print(f" - Height Variation (Std)  : {np.std(heights):.4f} m")
    print(f" - Max Torso Tilt Deviation: {np.max(tilts):.2f}° (Average: {np.mean(tilts):.2f}°)")
    print(f" - Mean Actuator Effort    : {np.mean(torques):.2f} Nm")
    print("==================================================================")

if __name__ == "__main__":
    evaluate_policy()
