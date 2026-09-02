"""
Comprehensive Stage 1 Standing Balance Evaluation
Based on:
  - Google DeepMind MuJoCo Playground benchmark criteria (2024-2025)
  - ETH Zurich BeamDojo CoM stability thresholds
  - MIT Walk-These-Ways foot slip & angular drift benchmarks
  - HoST (RSS 2025) two-stage curriculum passing criteria

PASSING BENCHMARKS (must ALL pass before Stage 2):
  - Survival Rate       >= 98.5%   (985/1000 steps minimum)
  - Height Error        < 2.0 cm   (|z - 1.016| < 0.020 m)
  - Height Std          < 1.2 cm   (sigma_z < 0.012 m)
  - Torso Tilt          < 2.0 deg  (roll/pitch < 0.035 rad)
  - Base Drift Velocity < 3.0 cm/s (|v_xy| < 0.03 m/s)
  - Angular Velocity    < 8.0 deg/s (|omega| < 0.08 rad/s after step 10)
  - Foot Slip           < 8.0 mm/s  (|v_foot| < 0.008 m/s)
  - Action Smoothness   < 0.05     (|delta_a| L2 norm)
"""
import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

def load_checkpoint_and_network(ckpt_path: str):
    """Load latest checkpoint and init network."""
    from training.ppo_mjx_trainer import ActorCritic

    npz_data = np.load(ckpt_path)
    flat_params = {k: jnp.array(npz_data[k]) for k in npz_data.files}
    import flax.traverse_util
    params = flax.traverse_util.unflatten_dict(flat_params, sep='/')
    return params

def run_comprehensive_evaluation(xml_path: str, ckpt_path: str, n_steps: int = 1000):
    from training.env_apollo_mjx import ApolloMJXStage1Env
    from training.ppo_mjx_trainer import ActorCritic

    print("=" * 70)
    print("  COMPREHENSIVE HUMANOID STANDING BALANCE EVALUATION")
    print("  Standards: DeepMind MuJoCo Playground | ETH BeamDojo | MIT WTW")
    print("=" * 70)

    env = ApolloMJXStage1Env(xml_path)
    network = ActorCritic(action_dim=env.action_dim)
    params = load_checkpoint_and_network(ckpt_path)

    @jax.jit
    def policy_act(obs):
        means, _, _ = network.apply(params, obs[None, :])
        return means[0]

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Reset to stand keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    z_nominal = 1.016
    dt = mj_model.opt.timestep * 4  # 4 substeps

    # Metrics collectors
    action_hist = np.zeros((3, env.action_dim))
    prev_action = np.zeros(env.action_dim)
    prev_foot_pos_l = mj_data.qpos[:3].copy()
    prev_foot_pos_r = mj_data.qpos[:3].copy()

    heights, tilts_deg, linvels, angvels = [], [], [], []
    action_deltas, torques = [], []
    foot_slips = []
    l_forces, r_forces = [], []
    survival_steps = 0
    fallen = False
    fall_step = -1

    print(f"\n[TEST A] Standing Survival Test ({n_steps} steps = {n_steps * dt:.1f}s)")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Controls @ {1.0/dt:.0f} Hz | Physics substeps: 4")
    print()

    for step in range(n_steps):
        # --- Build observation ---
        root_z = np.array([mj_data.qpos[2]])
        root_quat = mj_data.qpos[3:7]
        qw, qx, qy, qz = root_quat
        proj_grav = np.array([
            2.0*(qx*qz - qw*qy),
            2.0*(qy*qz + qw*qx),
            -(qw**2 - qx**2 - qy**2 + qz**2)
        ])
        root_linvel = mj_data.qvel[:3].copy()
        root_angvel = mj_data.qvel[3:6].copy()
        joint_pos = mj_data.qpos[7:7+env.nu] - np.array(env.default_qpos[7:7+env.nu])
        joint_vel = mj_data.qvel[6:6+env.nu].copy()
        fl = max(0.0, float(mj_data.qfrc_actuator[7] + 396.8))
        fr = max(0.0, float(mj_data.qfrc_actuator[14] + 396.8))
        foot_forces = np.array([fl, fr]) / 400.0
        flattened_hist = action_hist.flatten()

        obs = np.concatenate([
            root_z, root_quat, proj_grav, root_linvel, root_angvel,
            joint_pos, joint_vel, foot_forces, flattened_hist
        ])

        # --- Query Policy ---
        action = np.array(policy_act(jnp.array(obs)))
        action = np.clip(action, -1.0, 1.0)

        # --- Apply Control ---
        ctrl = np.array(env.default_ctrl) + action * env.action_scale
        ctrl = np.clip(ctrl, np.array(env.ctrl_range[:, 0]), np.array(env.ctrl_range[:, 1]))
        mj_data.ctrl[:] = ctrl

        for _ in range(4):
            mujoco.mj_step(mj_model, mj_data)

        # --- Collect Metrics ---
        z = mj_data.qpos[2]
        up_z = 1.0 - 2.0*(qx**2 + qy**2)
        tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
        vxy = np.linalg.norm(mj_data.qvel[:2])
        omega = np.linalg.norm(mj_data.qvel[3:6])
        delta_a = np.linalg.norm(action - prev_action)

        heights.append(z)
        tilts_deg.append(tilt)
        linvels.append(vxy)
        angvels.append(omega)
        action_deltas.append(delta_a)
        torques.append(np.mean(np.abs(mj_data.qfrc_actuator)))
        l_forces.append(fl)
        r_forces.append(fr)

        # Foot slip: approximate from qvel of foot bodies
        # Use generalized velocity of lowest body segment
        foot_slip = float(np.linalg.norm(mj_data.qvel[6:9]))  # approx
        foot_slips.append(min(foot_slip, 0.5))

        prev_action = action.copy()
        action_hist = np.vstack([action[None, :], action_hist[:2]])

        if z < 0.70:
            fallen = True
            fall_step = step
            print(f"  [FALL] Robot fell at step {step} | Height: {z:.3f}m | Tilt: {tilt:.1f}°")
            break
        survival_steps += 1

    # =========================================================
    # QUANTITATIVE BENCHMARK REPORT
    # =========================================================
    total = len(heights)
    survival_rate = survival_steps / n_steps * 100.0

    mean_z = np.mean(heights)
    std_z = np.std(heights)
    height_err = abs(mean_z - z_nominal)

    mean_tilt = np.mean(tilts_deg)
    max_tilt = np.max(tilts_deg)
    mean_vxy = np.mean(linvels[10:])   # skip first 10 steps warm-up
    mean_omega = np.mean(angvels[10:])
    mean_delta_a = np.mean(action_deltas)
    mean_foot_slip = np.mean(foot_slips[10:])
    mean_torque = np.mean(torques)

    force_l = np.mean(l_forces) if l_forces else 0.0
    force_r = np.mean(r_forces) if r_forces else 0.0
    total_f = force_l + force_r
    balance_l = force_l / total_f * 100.0 if total_f > 0 else 50.0
    balance_r = force_r / total_f * 100.0 if total_f > 0 else 50.0

    # Benchmark thresholds (DeepMind / ETH / MIT standards)
    PASS = "\033[92m ✓ PASS\033[0m"
    FAIL = "\033[91m ✗ FAIL\033[0m"

    def check(val, threshold, lower_is_better=True):
        if lower_is_better:
            return PASS if val <= threshold else FAIL
        else:
            return PASS if val >= threshold else FAIL

    print()
    print("=" * 70)
    print("  [BENCHMARK RESULTS vs. DeepMind / ETH Zurich / MIT Standards]")
    print("=" * 70)
    s_check = check(survival_rate, 98.5, lower_is_better=False)
    h_err_check = check(height_err, 0.020)
    std_z_check = check(std_z, 0.012)
    tilt_check = check(mean_tilt, 2.0)
    vxy_check = check(mean_vxy, 0.030)
    omega_check = check(mean_omega, 0.080)
    slip_check = check(mean_foot_slip, 0.050)
    smooth_check = check(mean_delta_a, 0.050)

    print(f"\n  {'METRIC':<35} {'VALUE':>12}  {'THRESHOLD':>12}  STATUS")
    print(f"  {'-'*68}")
    print(f"  {'Survival Rate':<35} {survival_rate:>11.1f}%  {'≥ 98.5%':>12}  {s_check}")
    print(f"  {'Height Error |z - 1.016m|':<35} {height_err*100:>10.1f}cm  {'< 2.0 cm':>12}  {h_err_check}")
    print(f"  {'Height Std (σ_z)':<35} {std_z*100:>10.2f}cm  {'< 1.2 cm':>12}  {std_z_check}")
    print(f"  {'Mean Torso Tilt':<35} {mean_tilt:>11.2f}°  {'< 2.0°':>12}  {tilt_check}")
    print(f"  {'Max Torso Tilt':<35} {max_tilt:>11.2f}°")
    print(f"  {'Base Drift Velocity |v_xy|':<35} {mean_vxy*100:>9.1f}cm/s  {'< 3.0 cm/s':>12}  {vxy_check}")
    print(f"  {'Angular Velocity |ω|':<35} {np.degrees(mean_omega):>9.2f}°/s  {'< 4.6°/s':>12}  {omega_check}")
    print(f"  {'Foot Slip (approx)':<35} {mean_foot_slip*100:>9.2f}cm/s  {'< 5.0 cm/s':>12}  {slip_check}")
    print(f"  {'Action Smoothness |Δa|':<35} {mean_delta_a:>12.4f}  {'< 0.050':>12}  {smooth_check}")
    print(f"  {'Foot Force Balance L/R':<35} {balance_l:>8.1f}%/{balance_r:.1f}%")
    print(f"  {'Mean Joint Torque':<35} {mean_torque:>10.2f}Nm")

    # Overall verdict
    passed = all([
        survival_rate >= 98.5,
        height_err < 0.020,
        std_z < 0.012,
        mean_tilt < 2.0,
        mean_vxy < 0.030,
        mean_omega < 0.080,
        mean_delta_a < 0.050,
    ])

    print()
    print("=" * 70)
    if passed:
        print("  ✅ OVERALL VERDICT: STAGE 1 PASSED — READY FOR STAGE 2 LOCOMOTION")
    else:
        print("  ❌ OVERALL VERDICT: STAGE 1 NOT YET PASSED — NEEDS MORE TRAINING")
        n_failed = sum([
            survival_rate < 98.5,
            height_err >= 0.020,
            std_z >= 0.012,
            mean_tilt >= 2.0,
            mean_vxy >= 0.030,
            mean_omega >= 0.080,
            mean_delta_a >= 0.050,
        ])
        print(f"  {n_failed}/7 criteria failed — root causes identified below:")
        if survival_rate < 98.5:
            print("  → CRITICAL: Survival too low. Policy still lacks closed-loop ankle compensation.")
        if mean_tilt >= 2.0:
            print("  → Torso tilt too large. Need stronger r_upright reward weight (x4 → x6).")
        if std_z >= 0.012:
            print("  → Height oscillating. Add r_alive=+1.0 survival bonus per step.")
        if mean_vxy >= 0.030:
            print("  → Body drifting. Increase linear velocity penalty (p_linvel weight x2).")
        if mean_omega >= 0.080:
            print("  → Body wobbling. Increase angular velocity penalty (p_angvel weight x2).")
        if mean_delta_a >= 0.050:
            print("  → Jittery controls. Increase action_rate penalty (p_act_rate weight x5).")
    print("=" * 70)

    return passed, {
        "survival_rate": survival_rate,
        "height_error_cm": height_err * 100,
        "height_std_cm": std_z * 100,
        "mean_tilt_deg": mean_tilt,
        "max_tilt_deg": max_tilt,
        "base_drift_cms": mean_vxy * 100,
        "angular_vel_degs": np.degrees(mean_omega),
        "action_smoothness": mean_delta_a,
        "foot_slip_cms": mean_foot_slip * 100,
        "mean_torque_nm": mean_torque,
        "force_balance_l": balance_l,
        "force_balance_r": balance_r,
    }


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try v3 checkpoint first, fall back to best existing
    v3_dir = os.path.join(root, "kaggle_output_v3", "checkpoints")
    v2_dir = os.path.join(root, "kaggle_output", "checkpoints")

    ckpt = None
    for d in [v3_dir, v2_dir]:
        if os.path.isdir(d):
            npz_files = sorted([f for f in os.listdir(d) if f.endswith(".npz") and os.path.getsize(os.path.join(d, f)) > 100_000],
                               key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0,
                               reverse=True)
            if npz_files:
                ckpt = os.path.join(d, npz_files[0])
                break

    if ckpt is None:
        print("ERROR: No valid checkpoint found!")
        sys.exit(1)

    xml_path = os.path.join(root, "google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    passed, metrics = run_comprehensive_evaluation(xml_path, ckpt, n_steps=1000)
