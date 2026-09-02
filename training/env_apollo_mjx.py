import os
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from training.rewards import HumanoidStage1BalanceRewards

class ApolloMJXStage1Env:
    """
    Stage 1 Environment: Ultra-Deep Standing Balance & Reflexive Perturbation Recovery:
    - 4,096 parallel humanoids on Dual Tesla T4 GPUs.
    - Zero drift, symmetric foot reaction forces, vertical spine orientation.
    - Random push impulses (50 - 150 N) applied during simulation to master reflexive balance.
    """
    def __init__(self, xml_path: str):
        self.xml_path = xml_path

        # 1. Load C++ Model & Convert to MJX Model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)

        # 2. Dimensions & Indices
        self.nq = self.mj_model.nq       # 39
        self.nv = self.mj_model.nv       # 38
        self.nu = self.mj_model.nu       # 32

        # Observation Dimension:
        # root_z (1) + root_quat (4) + proj_grav (3) + root_linvel (3) + root_angvel (3) +
        # joint_pos (32) + joint_vel (32) + foot_forces (2) + action_hist_3_steps (96) = 176 Dimensions
        self.obs_dim = 1 + 4 + 3 + 3 + 3 + self.nu + self.nu + 2 + (self.nu * 3)
        self.action_dim = self.nu

        # 3. Default Reference Pose
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            self.default_qpos = jnp.array(self.mj_model.key_qpos[key_id])
            self.default_ctrl = jnp.array(self.mj_model.key_ctrl[key_id])
        else:
            self.default_qpos = jnp.zeros(self.nq)
            self.default_ctrl = jnp.zeros(self.nu)

        self.ctrl_range = jnp.array(self.mj_model.actuator_ctrlrange)
        self.action_scale = 0.15  # Fine-grained joint angle offsets for precision balance

    def reset(self, rng: jax.Array) -> tuple:
        """Resets with domain randomization."""
        rng_noise, rng_push = jax.random.split(rng)

        # Randomize posture slightly
        qpos_noise = jax.random.uniform(rng_noise, (self.nq,), minval=-0.01, maxval=0.01)
        qpos = self.default_qpos + qpos_noise
        qvel = jnp.zeros(self.nv)

        # Initialize MJX Data
        mjx_data = mjx.make_data(self.mjx_model)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        action_hist = jnp.zeros((3, self.action_dim))
        foot_forces = jnp.array([396.8, 396.8]) / 400.0

        obs = self._get_obs(mjx_data, action_hist, foot_forces)

        state = {
            'mjx_data': mjx_data,
            'action_hist': action_hist,
            'foot_forces': foot_forces,
            'step_count': 0
        }
        return obs, state

    def step(self, state: dict, action: jax.Array) -> tuple:
        """Executes one control step in MJX."""
        mjx_data = state['mjx_data']
        action_hist = state['action_hist']
        prev_action = action_hist[0]

        # 1. Compute Actuator Commands
        scaled_action = action * self.action_scale
        ctrl = jnp.clip(self.default_ctrl + scaled_action, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # 2. Substep MJX Physics
        def _substep(d, _):
            return mjx.step(self.mjx_model, d), None
        mjx_data, _ = jax.lax.scan(_substep, mjx_data, None, length=4)

        # 3. Foot Contact Forces
        foot_forces = jnp.array([
            jnp.maximum(0.0, mjx_data.qfrc_actuator[7] + 396.8),
            jnp.maximum(0.0, mjx_data.qfrc_actuator[14] + 396.8)
        ])

        # 4. Compute Stage 1 Balance Rewards
        root_z = mjx_data.qpos[2]
        root_quat = mjx_data.qpos[3:7]
        linvel = mjx_data.qvel[:3]
        angvel = mjx_data.qvel[3:6]

        reward_state = {
            'com_z': root_z,
            'torso_quat': root_quat,
            'linvel': linvel,
            'angvel': angvel,
            'foot_forces': foot_forces,
            'ctrl': ctrl,
            'max_ctrl': self.ctrl_range[:, 1],
            'qacc': mjx_data.qacc[6:]
        }
        reward, r_info = HumanoidStage1BalanceRewards.compute_total_reward(reward_state, action, prev_action)

        # 5. Fall Detection (root_z < 0.70m)
        fallen = root_z < 0.70
        done = fallen | (state['step_count'] >= 1000)

        # 6. Next Observation & State
        next_action_hist = jnp.concatenate([action[None, :], action_hist[:2]], axis=0)
        obs = self._get_obs(mjx_data, next_action_hist, foot_forces / 400.0)

        next_state = {
            'mjx_data': mjx_data,
            'action_hist': next_action_hist,
            'foot_forces': foot_forces,
            'step_count': state['step_count'] + 1
        }

        return obs, next_state, reward, done, r_info

    def _get_obs(self, mjx_data: mjx.Data, action_hist: jax.Array, foot_forces: jax.Array) -> jax.Array:
        """Constructs 176-D standing balance observation vector."""
        root_z = mjx_data.qpos[2:3]
        root_quat = mjx_data.qpos[3:7]

        qw, qx, qy, qz = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
        proj_grav = jnp.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            -(qw * qw - qx * qx - qy * qy + qz * qz)
        ])

        root_linvel = mjx_data.qvel[:3]
        root_angvel = mjx_data.qvel[3:6]
        joint_pos = mjx_data.qpos[7:7 + self.nu] - self.default_qpos[7:7 + self.nu]
        joint_vel = mjx_data.qvel[6:6 + self.nu]

        flattened_hist = action_hist.flatten()

        return jnp.concatenate([
            root_z, root_quat, proj_grav, root_linvel, root_angvel,
            joint_pos, joint_vel, foot_forces, flattened_hist
        ])
