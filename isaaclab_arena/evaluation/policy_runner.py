# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import datetime
import os
import time
import torch
import tqdm
from gymnasium.wrappers import RecordVideo
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.camera_video import CameraObsVideoRecorder
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase


# ---------------------------------------------------------------------------
# Debug logger for inference
# ---------------------------------------------------------------------------


class DebugLogger:
    """Log per-step inference data: joint states, actions, and sampled camera frames.

    Output layout under out_dir/<timestamp>/:
      inference_log.csv   — one row per env.step()
      ep{N}_step{M}_<cam>.png — sampled frames every img_every steps
    """

    _FINGER_NAMES = ["fl_l", "fr_l", "fl_r", "fr_r"]

    def __init__(self, out_dir: str, img_every: int):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, ts)
        self.img_every = img_every
        os.makedirs(self.out_dir, exist_ok=True)

        self._csv_path = os.path.join(self.out_dir, "inference_log.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = None
        self._global_step = 0
        self._episode = 0
        self._ep_step = 0
        print(f"[DEBUG] inference log dir: {self.out_dir}")

    def new_episode(self):
        self._episode += 1
        self._ep_step = 0

    def log_step(self, env, actions: torch.Tensor, obs: dict):
        """Call after every env.step(). env must be the unwrapped IsaacLab env."""
        import re
        import numpy as np

        def _np(x):
            """Convert torch tensor, warp array, or numpy array to numpy."""
            if hasattr(x, "cpu"):                        # torch tensor
                return x.cpu().numpy()
            if hasattr(x, "numpy") and not isinstance(x, np.ndarray):  # warp array
                return x.numpy()
            return np.asarray(x)

        robot = env.scene["robot"]
        joint_pos = _np(robot.data.joint_pos)[0]   # convert full array, then take env 0
        joint_vel = _np(robot.data.joint_vel)[0]
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

        left_actual  = joint_pos[self._left_arm_ids]
        right_actual = joint_pos[self._right_arm_ids]
        fingers      = joint_pos[self._finger_ids]
        left_vel     = joint_vel[self._left_arm_ids]
        right_vel    = joint_vel[self._right_arm_ids]
        left_cmd     = cmd[:6]
        right_cmd    = cmd[6:12]
        left_grip    = cmd[12]
        right_grip   = cmd[13]

        row = {"episode": self._episode, "ep_step": self._ep_step, "global_step": self._global_step}
        for i in range(6):
            row[f"L_actual_j{i+1}"] = f"{left_actual[i]:.5f}"
            row[f"L_cmd_j{i+1}"]    = f"{left_cmd[i]:.5f}"
            row[f"L_err_j{i+1}"]    = f"{left_cmd[i] - left_actual[i]:.5f}"
            row[f"L_vel_j{i+1}"]    = f"{left_vel[i]:.5f}"
        for i in range(6):
            row[f"R_actual_j{i+1}"] = f"{right_actual[i]:.5f}"
            row[f"R_cmd_j{i+1}"]    = f"{right_cmd[i]:.5f}"
            row[f"R_err_j{i+1}"]    = f"{right_cmd[i] - right_actual[i]:.5f}"
            row[f"R_vel_j{i+1}"]    = f"{right_vel[i]:.5f}"
        row["L_grip_cmd"] = f"{left_grip:.3f}"
        row["R_grip_cmd"] = f"{right_grip:.3f}"
        for i, name in enumerate(self._FINGER_NAMES):
            row[f"finger_{name}"] = f"{fingers[i]:.5f}"

        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

        if self._global_step % self.img_every == 0:
            self._save_cameras(obs)

        self._global_step += 1
        self._ep_step += 1

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
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)
            if img.dtype != "uint8":
                img = (img * 255).clip(0, 255).astype("uint8")
            fname = f"ep{self._episode:04d}_step{self._ep_step:04d}_{cam_key}.png"
            Image.fromarray(img).save(os.path.join(self.out_dir, fname))

    def close(self):
        self._csv_file.close()
        print(f"[DEBUG] saved inference log → {self._csv_path}")
        print(f"[DEBUG] saved images        → {self.out_dir}/ep*.png")


# ---------------------------------------------------------------------------


def _export_stage_usd(path: str) -> None:
    """Export the current Isaac Sim stage to a USD file."""
    import omni.usd

    if not path.endswith((".usd", ".usda", ".usdc")):
        path += ".usd"
    stage = omni.usd.get_context().get_stage()
    # Flatten resolves all sublayer references into a single self-contained file.
    # Without flatten the file keeps relative/absolute references that must stay
    # accessible on the same machine. Use flatten=True when sharing the file.
    stage.Flatten().Export(path)
    print(f"[USD] Stage exported (flattened) → {path}")


def get_policy_cls(policy_type: str) -> type["PolicyBase"]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    from isaaclab_arena.assets.registries import PolicyRegistry

    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def is_distributed(args_cli: argparse.Namespace) -> bool:
    return (
        "cuda" in args_cli.device and hasattr(args_cli, "distributed") and args_cli.distributed and get_world_size() > 1
    )


def _warm_up_renderer(env, obs, num_warm_up_steps: int = 5):
    """Step the env with hold-position actions to let the renderer produce valid frames.

    The first few camera frames after reset can be black because the ray-tracing
    renderer needs time to converge. This runs a few steps commanding the robot
    to hold its current pose so the cameras stabilise before policy inference.
    """
    policy_obs = obs.get("policy", {})
    left_pos = policy_obs.get("left_joint_pos")
    right_pos = policy_obs.get("right_joint_pos")

    if left_pos is None or right_pos is None:
        # Fallback: step with zeros (robot holds default position via PD control)
        action_dim = env.unwrapped.action_manager.total_action_dim
        hold_action = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
    else:
        # Action format: [left_arm(6), right_arm(6), left_grip(1), right_grip(1)]
        # Gripper 0.0 = open (maintain current open state from reset)
        num_envs = left_pos.shape[0]
        hold_action = torch.cat([
            left_pos,
            right_pos,
            torch.zeros(num_envs, 1, device=left_pos.device),
            torch.zeros(num_envs, 1, device=left_pos.device),
        ], dim=1)

    for _ in range(num_warm_up_steps):
        obs, _, _, _, _ = env.step(hold_action)

    return obs


def rollout_policy(
    env,
    policy: "PolicyBase",
    num_steps: int | None,
    num_episodes: int | None,
    language_instruction: str | None = None,
    debug_logger: "DebugLogger | None" = None,
    num_warm_up_steps: int = 5,
) -> dict[str, Any]:
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    try:
        obs, _ = env.reset()
        if num_warm_up_steps > 0:
            obs = _warm_up_renderer(env, obs, num_warm_up_steps)
        policy.reset()
        if debug_logger:
            debug_logger.new_episode()
        # Determine language instruction: CLI/job-level override takes precedence over the task's own
        # description. Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing).
        task_description = language_instruction or env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description()
        policy.set_task_description(task_description)

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0
        t_sim_total_s = 0.0

        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)

                t0 = time.perf_counter()
                obs, _, terminated, truncated, _ = env.step(actions)
                t_sim_total_s += time.perf_counter() - t0

                if debug_logger:
                    debug_logger.log_step(env.unwrapped, actions, obs)

                if terminated.any() or truncated.any():
                    # Only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    if debug_logger:
                        debug_logger.new_episode()
                    # Break if number of episodes is reached
                    completed_episodes = env_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                    if num_episodes is not None:
                        pbar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
                            break
                # Break if number of steps is reached
                num_steps_completed += 1
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        pbar.close()

        n = max(num_steps_completed, 1)
        num_envs = env.unwrapped.num_envs
        server_time_s = getattr(policy, "_server_time_total_s", None)
        server_calls = getattr(policy, "_server_call_count", 0)
        env_steps = num_steps_completed * num_envs
        lines = [
            "=" * 60,
            "Timing Summary",
            "=" * 60,
            f"  num_envs             : {num_envs}",
            f"  Steps total          : {num_steps_completed}   env-steps: {env_steps}",
            f"  Simulation    total  : {t_sim_total_s:.2f} s"
            f"   avg {t_sim_total_s/n*1000:.1f} ms/step"
            f"   {t_sim_total_s/env_steps*1000:.1f} ms/env-step"
            f"   throughput {env_steps/t_sim_total_s:.1f} env-steps/s",
        ]
        if server_time_s is not None and server_calls > 0:
            lines += [
                f"  Server(infer+xfer)   : {server_time_s:.2f} s   avg {server_time_s/server_calls*1000:.1f} ms/call   calls={server_calls}",
            ]
        lines.append("=" * 60)
        tqdm.tqdm.write("\n".join(lines))

    except Exception as e:
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None


def main():
    """Run an IsaacLab Arena environment with a policy.
    Use --distributed with torchrun command for one process per GPU on multi-GPU machines. AppLauncher uses LOCAL_RANK for device.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    # Setting device to local rank before SimulationAppContext
    if is_distributed(args_cli):
        args_cli.device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}/{world_size}] One Isaac Lab instance per process on cuda:{local_rank}")

    with SimulationAppContext(args_cli):

        # Get the policy-type flag before proceeding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # Get the policy class from the policy type
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(
            f"[Rank {local_rank}/{world_size}] Requested policy type: {args_cli.policy_type} -> Policy class:"
            f" {policy_cls}"
        )

        # Add the example environment arguments + policy-related arguments to the parser
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"

        # Build scene. Use rgb_array render mode when recording so RecordVideo can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli)
        render_mode = "rgb_array" if args_cli.video else None
        env, cfg = arena_builder.make_registered_and_return_cfg(render_mode=render_mode)

        # Per-rank seed when distributed so each process has a different seed
        seed = args_cli.seed
        if seed is not None and is_distributed(args_cli):
            seed = seed + local_rank
        if seed is not None:
            set_seed(seed, env)

        # Create the policy from the arguments
        policy = policy_cls.from_args(args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")

        # Optionally wrap with RecordVideo and/or CameraObsVideoRecorder. The two flags
        # are independent: --video records the kit viewport (via env.render()),
        # --camera_video records the embodiment-mounted cameras (from obs["camera_obs"]).
        if args_cli.video or args_cli.camera_video:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            if num_steps is not None:
                video_length = num_steps
            else:
                # When num_episodes is set, capture exactly one episode's worth of frames.
                # max_episode_length is in environment steps, which matches our rollout cadence.
                video_length = num_episodes * env.unwrapped.max_episode_length

        if args_cli.video:
            env = RecordVideo(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step viewport video to:"
                f" {args_cli.video_dir}"
            )

        if args_cli.camera_video:
            # Record one mp4 per camera in obs["camera_obs"] (what the policy sees),
            # using the same encoder as RecordVideo.
            env = CameraObsVideoRecorder(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step per-camera videos to:"
                f" {args_cli.video_dir}"
            )

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")

        debug_logger = None
        if hasattr(args_cli, "debug") and args_cli.debug:
            debug_logger = DebugLogger(args_cli.debug_dir, args_cli.debug_img_every)

        if hasattr(args_cli, "export_usd") and args_cli.export_usd:
            _export_stage_usd(args_cli.export_usd)

        warm_up_steps = getattr(args_cli, "warm_up_steps", 5)
        metrics = rollout_policy(
            env, policy, num_steps, num_episodes, args_cli.language_instruction, debug_logger, warm_up_steps
        )

        if debug_logger:
            debug_logger.close()

        if metrics is not None:
            print(f"[Rank {local_rank}/{world_size}] Metrics: {metrics_to_plain_python_types(metrics)}")

        # NOTE(huikang, 2025-12-30)Explicitly clean up the remote policy client / server.
        # Do NOT rely on a __del__ destructor in policy for this, since destructors are
        # triggered implicitly and their execution time (or even whether they run)
        # is not guaranteed, which makes resource cleanup unreliable.
        if policy.is_remote:
            policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
