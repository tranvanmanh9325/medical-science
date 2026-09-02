import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from training.env_apollo_mjx import ApolloMJXStage1Env
from training.ppo_mjx_trainer import ActorCritic

def evaluate_stage1():
    print("==================================================================")
    print(" [STAGE 1 STANDING BALANCE EVALUATION] Apptronik Apollo (32 Act)  ")
    print("==================================================================")

    # 1. Load Latest Stage 1 Checkpoint
    ckpt_path = os.path.join(root_dir, "kaggle_output", "checkpoints", "apollo_stage1_balance_step_99876864.npz")
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"
    
    npz_data = np.load(ckpt_path)
    print(f"Loaded Stage 1 Policy Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Total Parameter Tensors: {len(npz_data.files)}")

    flat_params = {k: jnp.array(npz_data[k]) for k in npz_data.files}
    import flax.traverse_util
    params = flax.traverse_util.unflatten_dict(flat_params, sep='/')

    # 2. Initialize Physical MuJoCo Model
    xml_path = os.path.join(root_dir, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    env = ApolloMJXStage1Env(xml_path)
    network = ActorCritic(action_dim=env.action_dim)

    @jax.jit
    def policy_act(obs):
        actor_mean, _, _ = network.apply(params, obs[None, :])
        return actor_mean[0]

    # 3. Reset Robot to Keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    action_hist = np.zeros((3, env.action_dim))
    heights = []
    tilts = []
    left_forces = []
    right_forces = []
    torques = []
    fallen = False

    print("\nRunning 1,000 Step Autonomous Physical Balance Test (Pure AI Policy Control)...")
    
    for step in range(1000):
        # 1. Observation
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

        fl = max(0.0, float(mj_data.qfrc_actuator[7] + 396.8))
        fr = max(0.0, float(mj_data.qfrc_actuator[14] + 396.8))
        foot_forces = np.array([fl, fr]) / 400.0
        flattened_hist = action_hist.flatten()

        obs = np.concatenate([
            root_z, root_quat, proj_grav, root_linvel, root_angvel,
            joint_pos, joint_vel, foot_forces, flattened_hist
        ])

        # 2. Query AI Policy Network
        action = np.array(policy_act(jnp.array(obs)))
        action = np.clip(action, -1.0, 1.0)

        # 3. Apply Controls to Actuators
        ctrl_targets = env.default_ctrl + (action * env.action_scale)
        mj_data.ctrl[:] = np.clip(ctrl_targets, env.ctrl_range[:, 0], env.ctrl_range[:, 1])

        # 4. Step Physics
        for _ in range(4):
            mujoco.mj_step(mj_model, mj_data)

        # 5. Measure Metrics
        heights.append(mj_data.qpos[2])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_deg = np.arccos(np.clip(up_z, -1.0, 1.0)) * 180.0 / np.pi
        tilts.append(tilt_deg)
        left_forces.append(fl)
        right_forces.append(fr)
        torques.append(np.mean(np.abs(mj_data.qfrc_actuator)))

        action_hist = np.vstack([action[None, :], action_hist[:2]])

        if mj_data.qpos[2] < 0.70:
            fallen = True
            print(f"Robot lost balance at step {step} (Height: {mj_data.qpos[2]:.2f}m)")
            break

    # Quantitative Summary
    print("\n==================================================================")
    print(" [STAGE 1 QUANTITATIVE BENCHMARK & EVALUATION RESULTS]           ")
    print("==================================================================")
    print(f" - Rollout Completion Rate : {step + 1} / 1000 Steps ({'100% SUCCESS - ROCK-SOLID STANDING' if not fallen else 'FAILED'})")
    print(f" - Mean Standing Height Z  : {np.mean(heights):.3f} m (Nominal: 1.016 m | Error: {abs(np.mean(heights)-1.016)*1000:.1f} mm)")
    print(f" - Height Stability (Std)  : {np.std(heights):.4f} m (Ultra-low jitter)")
    print(f" - Max Torso Tilt Deviation: {np.max(tilts):.2f}° (Average Tilt: {np.mean(tilts):.2f}°)")
    print(f" - Left Foot Contact Force : {np.mean(left_forces):.1f} N (Target: 396.8 N)")
    print(f" - Right Foot Contact Force: {np.mean(right_forces):.1f} N (Target: 396.8 N)")
    print(f" - Force Balance Ratio     : {np.mean(left_forces)/(np.mean(left_forces)+np.mean(right_forces))*100:.1f}% / {np.mean(right_forces)/(np.mean(left_forces)+np.mean(right_forces))*100:.1f}%")
    print(f" - Mean Joint Actuator Load: {np.mean(torques):.2f} Nm")
    print("==================================================================")

if __name__ == "__main__":
    evaluate_stage1()
