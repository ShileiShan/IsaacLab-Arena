# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record an mp4 video of the rollout (uses gymnasium.wrappers.RecordVideo).",
    )
    parser.add_argument(
        "--video_dir",
        "--video-dir",
        type=str,
        default="/eval/videos",
        help="Output directory for recorded videos. Created if missing. Used with --video and/or --camera_video.",
    )
    parser.add_argument(
        "--camera_video",
        "--camera-video",
        action="store_true",
        default=False,
        help=(
            "Record one mp4 per camera in obs['camera_obs'] (what the policy actually sees)."
            " Independent of --video; use either or both."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Enable debug mode: log per-step joint positions, commanded actions, policy server "
            "request/response, and sampled camera frames to --debug_dir."
        ),
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="inference_debug",
        help="Output directory for debug logs and images (default: inference_debug/).",
    )
    parser.add_argument(
        "--debug_img_every",
        type=int,
        default=10,
        metavar="N",
        help="Save a camera frame every N steps in debug mode (default: 10).",
    )
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Number of hold-position steps after reset before policy inference starts. "
            "Allows the renderer to produce valid camera frames (default: 5)."
        ),
    )
    parser.add_argument(
        "--speed_log_interval",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Update the progress bar with real-time factor every N environment steps. "
            "Set to 0 to disable periodic speed reporting (default: 100)."
        ),
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
