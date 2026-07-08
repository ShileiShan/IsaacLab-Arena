# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--loop",
    action="store_true",
    default=False,
    help="Loop replay indefinitely instead of exiting after all episodes are played.",
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--robot_rot_xyzw",
    type=float,
    nargs=4,
    default=(0.0, 0.0, 0.0, 1.0),
    metavar=("X", "Y", "Z", "W"),
    help=(
        "Override the robot base rotation quaternion (xyzw) at every episode reset. "
        "Example: --robot_rot_xyzw 0 0 1 0  (180° around Z). "
        "If omitted, the rotation stored in the HDF5 root_pose is used as-is."
    ),
)
parser.add_argument(
    "--record_cameras",
    action="store_true",
    default=False,
    help=(
        "Record camera observations (RGB) during replay and save to a new HDF5 file "
        "with '_cam' suffix. Requires --enable_cameras in the environment config."
    ),
)
parser.add_argument(
    "--record_cameras_resize",
    type=int,
    nargs=2,
    default=None,
    metavar=("H", "W"),
    help=(
        "Resize camera images before saving (height width). "
        "If omitted, images are saved at native resolution (480x640)."
    ),
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help=(
        "Enable debug mode: write per-step joint/action/gripper data to a CSV log and save "
        "sampled camera frames as PNG images under --debug_dir."
    ),
)
parser.add_argument(
    "--debug_dir",
    type=str,
    default="replay_debug",
    help="Output directory for debug logs and images (default: replay_debug/).",
)
parser.add_argument(
    "--debug_img_every",
    type=int,
    default=10,
    metavar="N",
    help="Save a camera frame every N steps in debug mode (default: 10).",
)
parser.add_argument(
    "--export_usd",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Export the fully-initialised simulation stage as a USD file at the given path "
        "(e.g. /tmp/scene.usd). Useful for inspecting the scene in Isaac Sim."
    ),
)
# Add the example environments CLI args
# NOTE(alexmillane, 2025.09.04): This has to be added last, because
# of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import csv
import datetime
import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

class DebugLogger:
    """Writes per-step joint/action/gripper data to CSV and saves camera frames."""

    # CSV column order — must match _build_row()
    _JOINT_NAMES = [f"j{i+1}" for i in range(6)]
    _FINGER_NAMES = ["fl_l", "fr_l", "fl_r", "fr_r"]
    _ARM_CMD_NAMES = [f"cmd_j{i+1}" for i in range(6)]

    def __init__(self, out_dir: str, img_every: int):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, ts)
        self.img_every = img_every
        os.makedirs(self.out_dir, exist_ok=True)

        self._csv_path = os.path.join(self.out_dir, "replay_log.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = None  # initialised on first row (column count depends on obs)
        self._step = 0
        self._episode = 0
        print(f"[DEBUG] log dir: {self.out_dir}")

    # ------------------------------------------------------------------
    def new_episode(self, episode_idx: int):
        self._episode = episode_idx
        self._step = 0

    # ------------------------------------------------------------------
    def log_step(self, env, actions: torch.Tensor, obs: dict):
        """Call after every env.step()."""
        import re
        import numpy as np

        def _np(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
                return x.numpy()
            return np.asarray(x)

        robot = env.scene["robot"]

        # --- joint positions (actual) ---
        joint_pos = _np(robot.data.joint_pos)[0]   # (16,)
        joint_vel = _np(robot.data.joint_vel)[0]   # (16,)

        # --- commanded action (14-dim arena format) ---
        cmd = _np(actions)[0]                       # (14,)

        # Resolve joint indices by name (USD joint order may differ from assumed layout)
        if not hasattr(self, "_joint_indices_resolved"):
            names = robot.data.joint_names
            self._left_arm_ids = sorted(
                [i for i, n in enumerate(names) if re.match(r"^joint[1-6]_l$", n)],
                key=lambda i: names[i],
            )
            self._right_arm_ids = sorted(
                [i for i, n in enumerate(names) if re.match(r"^joint[1-6]_r$", n)],
                key=lambda i: names[i],
            )
            self._finger_ids = [
                i for i, n in enumerate(names) if re.match(r"^finger_joint.*", n)
            ]
            self._joint_indices_resolved = True

        # --- compute per-joint error: cmd arm - actual arm ---
        left_arm_actual  = joint_pos[self._left_arm_ids]
        right_arm_actual = joint_pos[self._right_arm_ids]
        fingers_actual   = joint_pos[self._finger_ids]
        left_arm_cmd     = cmd[:6]
        right_arm_cmd    = cmd[6:12]
        left_grip_cmd    = cmd[12]
        right_grip_cmd   = cmd[13]
        left_err         = left_arm_cmd  - left_arm_actual
        right_err        = right_arm_cmd - right_arm_actual
        left_vel         = joint_vel[self._left_arm_ids]
        right_vel        = joint_vel[self._right_arm_ids]

        row = {
            "episode": self._episode,
            "step": self._step,
        }
        for i in range(6):
            row[f"L_actual_j{i+1}"]  = f"{left_arm_actual[i]:.5f}"
            row[f"L_cmd_j{i+1}"]     = f"{left_arm_cmd[i]:.5f}"
            row[f"L_err_j{i+1}"]     = f"{left_err[i]:.5f}"
            row[f"L_vel_j{i+1}"]     = f"{left_vel[i]:.5f}"
        for i in range(6):
            row[f"R_actual_j{i+1}"]  = f"{right_arm_actual[i]:.5f}"
            row[f"R_cmd_j{i+1}"]     = f"{right_arm_cmd[i]:.5f}"
            row[f"R_err_j{i+1}"]     = f"{right_err[i]:.5f}"
            row[f"R_vel_j{i+1}"]     = f"{right_vel[i]:.5f}"
        row["L_grip_cmd"]   = f"{left_grip_cmd:.1f}"
        row["R_grip_cmd"]   = f"{right_grip_cmd:.1f}"
        for i, name in enumerate(self._FINGER_NAMES):
            row[f"finger_{name}"] = f"{fingers_actual[i]:.5f}"

        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

        # --- camera frames ---
        if self._step % self.img_every == 0:
            self._save_cameras(obs)

        self._step += 1

    # ------------------------------------------------------------------
    def _save_cameras(self, obs: dict):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return

        def _np(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
                return x.numpy()
            return np.asarray(x)

        cam_obs = obs.get("camera_obs", {})
        for cam_key, tensor in cam_obs.items():
            img = _np(tensor)[0]
            fname = f"ep{self._episode:04d}_step{self._step:04d}_{cam_key}.png"
            Image.fromarray(img).save(os.path.join(self.out_dir, fname))

    # ------------------------------------------------------------------
    def close(self):
        self._csv_file.close()
        print(f"[DEBUG] saved log to {self._csv_path}")


# ---------------------------------------------------------------------------
# Camera HDF5 recorder
# ---------------------------------------------------------------------------


class CameraHDF5Recorder:
    """Records camera observations during replay and saves to HDF5."""

    CAMERA_KEYS = ["first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb"]

    def __init__(self, source_path: str, resize_hw: tuple | None = None):
        import h5py
        import numpy as np

        self._h5py = h5py
        self._np = np
        self._resize_hw = resize_hw

        base, ext = os.path.splitext(source_path)
        self._output_path = f"{base}_cam{ext}"
        self._file = h5py.File(self._output_path, "w")
        self._file.attrs["format_version"] = 1
        self._data_grp = self._file.create_group("data")
        self._episode_idx = -1
        self._buffers: dict = {}
        print(f"[CameraRecorder] output: {self._output_path}")

    def new_episode(self, episode_index: int):
        self._flush()
        self._episode_idx = episode_index
        self._buffers = {k: [] for k in self.CAMERA_KEYS}

    def record_step(self, obs: dict, env_id: int = 0):
        cam_obs = obs.get("camera_obs", {})
        if not cam_obs:
            return
        for key in self.CAMERA_KEYS:
            if key not in cam_obs:
                continue
            frame = cam_obs[key][env_id]
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            if self._resize_hw is not None:
                frame = self._resize(frame, self._resize_hw)
            self._buffers.setdefault(key, []).append(frame)

    def _resize(self, img, hw: tuple):
        import cv2

        h, w = hw
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _flush(self):
        if self._episode_idx < 0 or not self._buffers:
            return
        np = self._np
        demo_grp = self._data_grp.create_group(f"demo_{self._episode_idx}")
        cam_grp = demo_grp.create_group("camera_obs")
        for key, frames in self._buffers.items():
            if not frames:
                continue
            arr = np.stack(frames, axis=0).astype(np.uint8)
            cam_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        num_steps = len(next(iter(self._buffers.values()), []))
        demo_grp.attrs["num_samples"] = num_steps
        self._buffers = {}

    def close(self):
        self._flush()
        total = sum(self._data_grp[k].attrs.get("num_samples", 0) for k in self._data_grp.keys())
        self._data_grp.attrs["total"] = total
        self._file.close()
        print(f"[CameraRecorder] saved {self._output_path}")


# ---------------------------------------------------------------------------


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()

    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = list(args_cli.select_episodes) if args_cli.select_episodes else list(range(episode_count))

    num_envs = args_cli.num_envs

    # Compile an IsaacLab compatible arena environment configuration
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped

    # `env.reset_to()` below drives scene.reset_to(), which unconditionally writes a
    # zero velocity to every rigid object in the recorded state -- including kinematic
    # ones, which PhysX rejects. See disable_kinematic_rigid_object_velocity_writes().
    from isaaclab_arena.utils.phyx_utils import disable_kinematic_rigid_object_velocity_writes

    disable_kinematic_rigid_object_velocity_writes(env)

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # Debug logger
    debug_logger = DebugLogger(args_cli.debug_dir, args_cli.debug_img_every) if args_cli.debug else None

    # Camera recorder
    cam_recorder = None
    if args_cli.record_cameras:
        resize_hw = tuple(args_cli.record_cameras_resize) if args_cli.record_cameras_resize else None
        cam_recorder = CameraHDF5Recorder(args_cli.dataset_file, resize_hw=resize_hw)

    # reset before starting
    obs, _ = env.reset()
    teleop_interface.reset()

    # Export stage after first reset so all objects are placed
    if args_cli.export_usd:
        import omni.usd
        usd_path = args_cli.export_usd
        if not usd_path.endswith((".usd", ".usda", ".usdc")):
            usd_path += ".usd"
        omni.usd.get_context().get_stage().Flatten().Export(usd_path)
        print(f"[USD] Stage exported (flattened) → {usd_path}")

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    base_episode_indices = args_cli.select_episodes if args_cli.select_episodes else list(range(episode_count))
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is None and args_cli.loop:
                            episode_indices_to_replay = list(base_episode_indices)
                            next_episode_index = episode_indices_to_replay.pop(0)

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            # Optionally override the robot base rotation from CLI
                            if args_cli.robot_rot_xyzw is not None:
                                rot = torch.tensor(args_cli.robot_rot_xyzw, dtype=torch.float32, device=env.device)
                                initial_state["articulation"]["robot"]["root_pose"][0, 3:] = rot
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            if debug_logger:
                                debug_logger.new_episode(next_episode_index)
                            if cam_recorder:
                                cam_recorder.new_episode(next_episode_index)
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                obs, _, _, _, _ = env.step(actions)

                if debug_logger:
                    debug_logger.log_step(env, actions, obs)

                if cam_recorder:
                    cam_recorder.record_step(obs)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            if not args_cli.loop:
                break

    if debug_logger:
        debug_logger.close()

    if cam_recorder:
        cam_recorder.close()

    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
