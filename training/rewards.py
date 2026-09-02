import jax
import jax.numpy as jnp

class HumanoidStage1BalanceRewards:
    """
    Google DeepMind Stage 1: Ultra-Deep Standing Balance & Reflexive Push Recovery:
    - Precision height holding (z_CoM = 1.016m).
    - Spine/Torso absolute vertical orientation (theta_tilt < 1.5°).
    - Bilateral 50/50 foot ground reaction force distribution.
    - Zero linear & angular base drift.
    - Low-energy motor torque regularization.
    """
    @staticmethod
    def reward_height(com_z: jnp.ndarray, target_z: float = 1.016, sigma: float = 0.04) -> jnp.ndarray:
        """Holds exact anatomical standing pelvis height."""
        height_err = jnp.square(com_z - target_z)
        return jnp.exp(-height_err / (2.0 * sigma * sigma))

    @staticmethod
    def reward_upright_torso(torso_quat: jnp.ndarray, sigma: float = 0.05) -> jnp.ndarray:
        """Enforces vertical spinal posture with zero tilt."""
        qw, qx, qy, qz = torso_quat[0], torso_quat[1], torso_quat[2], torso_quat[3]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_err = jnp.square(1.0 - up_z)
        return jnp.exp(-tilt_err / (2.0 * sigma * sigma))

    @staticmethod
    def reward_zero_drift(linvel: jnp.ndarray, angvel: jnp.ndarray, sigma: float = 0.10) -> jnp.ndarray:
        """Penalizes any translation or wobbling during standing."""
        drift_sq = jnp.sum(jnp.square(linvel)) + 0.5 * jnp.sum(jnp.square(angvel))
        return jnp.exp(-drift_sq / (2.0 * sigma * sigma))

    @staticmethod
    def reward_foot_symmetry(foot_forces: jnp.ndarray, target_force_per_foot: float = 396.8, sigma: float = 80.0) -> jnp.ndarray:
        """Enforces equal 50/50 bilateral weight distribution across both feet."""
        force_err = jnp.sum(jnp.square(foot_forces - target_force_per_foot))
        return jnp.exp(-force_err / (2.0 * sigma * sigma))

    @staticmethod
    def penalty_joint_torques(ctrl: jnp.ndarray, max_ctrl: jnp.ndarray) -> jnp.ndarray:
        """Minimizes motor effort for long-term standing stamina."""
        normalized_ctrl = ctrl / jnp.maximum(1e-3, max_ctrl)
        return jnp.sum(jnp.square(normalized_ctrl))

    @staticmethod
    def penalty_action_rate(action: jnp.ndarray, prev_action: jnp.ndarray) -> jnp.ndarray:
        """Penalizes rapid jittery control signals."""
        return jnp.sum(jnp.square(action - prev_action))

    @classmethod
    def compute_total_reward(cls, state: dict, action: jnp.ndarray, prev_action: jnp.ndarray) -> tuple:
        """Computes composite Stage 1 Standing Balance reward."""
        r_h = cls.reward_height(state['com_z'])
        r_up = cls.reward_upright_torso(state['torso_quat'])
        r_drift = cls.reward_zero_drift(state['linvel'], state['angvel'])
        r_sym = cls.reward_foot_symmetry(state.get('foot_forces', jnp.array([396.8, 396.8])))

        p_torque = cls.penalty_joint_torques(state['ctrl'], state['max_ctrl'])
        p_act_rate = cls.penalty_action_rate(action, prev_action)
        p_acc = jnp.sum(jnp.square(state.get('qacc', jnp.zeros(32))))

        total = (
            3.0 * r_h +
            3.0 * r_up +
            2.0 * r_drift +
            2.0 * r_sym -
            0.003 * p_torque -
            0.02 * p_act_rate -
            0.0001 * p_acc
        )

        breakdown = {
            'r_height': r_h,
            'r_upright': r_up,
            'r_drift': r_drift,
            'r_sym': r_sym,
            'total': total
        }

        return total, breakdown
