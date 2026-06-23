"""Convert real Piper robot HDF5 data to IsaacLab Arena replay format.

Real robot HDF5 format:
  action:               (T, 14) — [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  observations/qpos:    (T, 14) — same layout as action
  observations/qvel:    (T, 14)

IsaacLab Arena HDF5 format:
  data/
    demo_N/
      actions:                                  (T, 14) Arena action format
      initial_state/
        articulation/robot/joint_position:      (16,)
        articulation/robot/joint_velocity:      (16,)
        articulation/robot/root_pose:           (7,)  xyzw
        articulation/robot/root_velocity:       (6,)
  data.attrs["env_args"] = '{"env_name": "pick_and_place_piper"}'

Arena action layout (14 dims):
  [0:6]  left_arm  (JointPositionAction, radians)
  [6:12] right_arm (JointPositionAction, radians)
  [12]   left_gripper_cmd  (BinaryJointPosition: 1.0=open, 0.0=close)
  [13]   right_gripper_cmd (BinaryJointPosition: 1.0=open, 0.0=close)

Arena robot joint_position (16 dims):
  [0:6]  joint1_l ... joint6_l
  [6:12] joint1_r ... joint6_r
  [12]   finger_joint_left_l   (open= 0.035, close=0.0)
  [13]   finger_joint_right_l  (open=-0.035, close=0.0)
  [14]   finger_joint_left_r   (open= 0.035, close=0.0)
  [15]   finger_joint_right_r  (open=-0.035, close=0.0)

Usage:
  python convert_real_piper_hdf5.py \
    --input /path/to/real_data.hdf5 \
    --output /path/to/arena_data.hdf5 \
    [--gripper_threshold 0.01]   # meters, above this = open
"""

import argparse
import json
import numpy as np
import h5py

# Gripper physical range on Piper: fully open = 0.035 m, closed = 0.0 m
GRIPPER_OPEN_M = 0.035
GRIPPER_OPEN_SIM = 0.035    # finger_joint_left value when open
GRIPPER_CLOSE_SIM = 0.0

# Robot base orientation in simulation (xyzw quaternion).
# Must match _DEFAULT_ROT_XYZW in double_piper.py so replay and inference see the same pose.
# 180° around Z — robot faces +X toward the table.
_ROOT_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)


def real_qpos_to_sim_joint_position(qpos: np.ndarray, gripper_threshold: float) -> np.ndarray:
    """Convert real 14-dim qpos to 16-dim sim joint_position.

    qpos layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    sim layout:  [left_arm(6), right_arm(6),
                  finger_joint_left_l, finger_joint_right_l,
                  finger_joint_left_r, finger_joint_right_r]
    """
    left_arm = qpos[:6]
    left_grip = qpos[6]
    right_arm = qpos[7:13]
    right_grip = qpos[13]

    # Simulate symmetric gripper fingers
    left_open = float(left_grip > gripper_threshold)
    right_open = float(right_grip > gripper_threshold)

    finger_left_l = GRIPPER_OPEN_SIM * left_open
    finger_right_l = -GRIPPER_OPEN_SIM * left_open
    finger_left_r = GRIPPER_OPEN_SIM * right_open
    finger_right_r = -GRIPPER_OPEN_SIM * right_open

    return np.concatenate([
        left_arm, right_arm,
        [finger_left_l, finger_right_l, finger_left_r, finger_right_r],
    ]).astype(np.float32)


def real_action_to_arena_action(action: np.ndarray, gripper_threshold: float) -> np.ndarray:
    """Convert real 14-dim action to Arena 14-dim action.

    Real:  [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    Arena: [left_arm(6), right_arm(6), left_grip_cmd(1), right_grip_cmd(1)]
    """
    left_arm = action[:6]
    left_grip = action[6]
    right_arm = action[7:13]
    right_grip = action[13]

    # BinaryJointPositionZeroToOneAction: 0.0 = OPEN, 1.0 = CLOSE
    # Real robot: high value (~0.09) = OPEN, low/negative (~-0.005) = CLOSE
    left_cmd = 0.0 if left_grip > gripper_threshold else 1.0
    right_cmd = 0.0 if right_grip > gripper_threshold else 1.0

    return np.concatenate([left_arm, right_arm, [left_cmd, right_cmd]]).astype(np.float32)


def convert(input_path: str, output_path: str, gripper_threshold: float, env_name: str):
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        data_grp = dst.create_group("data")
        data_grp.attrs["env_args"] = json.dumps({"env_name": env_name})
        data_grp.attrs["format_version"] = 1  # XYZW quaternion format

        # Determine episode structure
        # Try ALOHA-style: top-level action/observations
        if "action" in src:
            episodes = {"demo_0": src}
            print("Single-episode file detected.")
        else:
            # Multi-episode: src["data"]["demo_N"]
            episodes = {k: src["data"][k] for k in src["data"].keys()}
            print(f"Multi-episode file: {len(episodes)} episodes.")

        for ep_idx, (ep_key, ep_grp) in enumerate(episodes.items()):
            demo_name = f"demo_{ep_idx}"
            print(f"  Converting {ep_key} -> {demo_name}")

            actions_raw = np.array(ep_grp["action"])      # (T, 14)
            qpos = np.array(ep_grp["observations/qpos"])  # (T, 14)
            qvel_raw = np.array(ep_grp["observations/qvel"])  # (T, 14)

            T = actions_raw.shape[0]

            # Convert actions
            arena_actions = np.stack([
                real_action_to_arena_action(actions_raw[t], gripper_threshold)
                for t in range(T)
            ])  # (T, 14)

            # Initial state from first frame of qpos
            # All initial_state arrays need a batch dimension (1, N)
            init_joint_pos = real_qpos_to_sim_joint_position(qpos[0], gripper_threshold)
            init_joint_pos = init_joint_pos[np.newaxis, :]   # (1, 16)
            init_joint_vel = np.zeros((1, 16), dtype=np.float32)

            # Robot at world origin with default sim orientation (matches _DEFAULT_ROT_XYZW in double_piper.py)
            root_pose = np.array([[0.0, 0.0, 0.1, *_ROOT_ROT_XYZW]], dtype=np.float32)  # (1, 7) xyz+xyzw
            root_vel = np.zeros((1, 6), dtype=np.float32)  # (1, 6)

            # Write demo group
            demo_grp = data_grp.create_group(demo_name)
            demo_grp.create_dataset("actions", data=arena_actions)

            init_grp = demo_grp.create_group("initial_state")
            robot_grp = init_grp.create_group("articulation/robot")
            robot_grp.create_dataset("joint_position", data=init_joint_pos)
            robot_grp.create_dataset("joint_velocity", data=init_joint_vel)
            robot_grp.create_dataset("root_pose", data=root_pose)
            robot_grp.create_dataset("root_velocity", data=root_vel)

        print(f"Done. Written {len(episodes)} episodes to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert real Piper HDF5 to IsaacLab Arena format")
    parser.add_argument("--input", required=True, help="Input real robot HDF5 file")
    parser.add_argument("--output", required=True, help="Output Arena HDF5 file")
    parser.add_argument(
        "--gripper_threshold", type=float, default=0.01,
        help="Gripper open threshold in meters (default: 0.01). Above this = open."
    )
    parser.add_argument(
        "--env_name", type=str, default="pick_and_place_piper",
        help="Arena environment name (default: pick_and_place_piper)"
    )
    args = parser.parse_args()

    convert(args.input, args.output, args.gripper_threshold, args.env_name)


if __name__ == "__main__":
    main()
