# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.evaluation.double_piper_single_object_eval_runner import (
    _finalize_object_summaries,
    build_jobs_from_config,
    save_eval_log,
)


def test_double_piper_single_object_eval_config_expands_objects_and_positions():
    config = {
        "policy_type": "zero_action",
        "num_episodes": 2,
        "num_envs": 3,
        "arena_env_args": {
            "environment": "pick_and_place_piper",
            "embodiment": "double_piper_abs_joint_pos",
        },
        "positions": [
            {"name": "center", "bin_offset_xyz": [0.0, 0.0, 0.045]},
            {"name": "left", "bin_offset_xyz": [0.0, -0.05, 0.045]},
        ],
        "objects": [
            "rubiks_cube_hot3d_robolab",
            {"name": "tomato_soup_can"},
        ],
    }

    jobs, metadata = build_jobs_from_config(config)

    assert len(jobs) == 4
    assert jobs[0].num_envs == 3
    assert jobs[0].num_episodes == 2
    assert jobs[0].arena_env_args == [
        "--num_envs",
        "3",
        "pick_and_place_piper",
        "--embodiment",
        "double_piper_abs_joint_pos",
        "--pick_up_object",
        "rubiks_cube_hot3d_robolab",
        "--pick_object_bin_offset_xyz",
        "0.0",
        "0.0",
        "0.045",
    ]
    assert metadata[jobs[1].name]["object"] == "rubiks_cube_hot3d_robolab"
    assert metadata[jobs[1].name]["position"] == "left"


def test_double_piper_single_object_eval_config_encodes_boolean_optional_false():
    config = {
        "policy_type": "zero_action",
        "num_steps": 1,
        "arena_env_args": {
            "environment": "pick_and_place_piper",
            "use_tiled_camera": False,
        },
        "positions": [[0.0, 0.0, 0.045]],
        "objects": ["rubiks_cube_hot3d_robolab"],
    }

    jobs, _ = build_jobs_from_config(config)

    assert "--use_tiled_camera" not in jobs[0].arena_env_args
    assert "--no-use_tiled_camera" in jobs[0].arena_env_args


def test_double_piper_single_object_eval_summarizes_object_and_overall_success_rates(tmp_path):
    results = {
        "config_path": "test.yaml",
        "summary": {},
        "jobs": {
            "cube_center": {
                "object": "cube",
                "position": "center",
                "bin_offset_xyz": [0.0, 0.0, 0.045],
                "status": "completed",
                "metrics": {"success_rate": 1.0},
            },
            "cube_left": {
                "object": "cube",
                "position": "left",
                "bin_offset_xyz": [0.0, -0.05, 0.045],
                "status": "completed",
                "metrics": {"success_rate": 0.5},
            },
            "can_center": {
                "object": "can",
                "position": "center",
                "bin_offset_xyz": [0.0, 0.0, 0.045],
                "status": "completed",
                "metrics": {"success_rate": 0.0},
            },
        },
        "objects": {
            "cube": {
                "positions": [
                    {
                        "job": "cube_center",
                        "object": "cube",
                        "position": "center",
                        "bin_offset_xyz": [0.0, 0.0, 0.045],
                        "status": "completed",
                        "metrics": {"success_rate": 1.0},
                    },
                    {
                        "job": "cube_left",
                        "object": "cube",
                        "position": "left",
                        "bin_offset_xyz": [0.0, -0.05, 0.045],
                        "status": "completed",
                        "metrics": {"success_rate": 0.5},
                    },
                ],
            },
            "can": {
                "positions": [
                    {
                        "job": "can_center",
                        "object": "can",
                        "position": "center",
                        "bin_offset_xyz": [0.0, 0.0, 0.045],
                        "status": "completed",
                        "metrics": {"success_rate": 0.0},
                    }
                ],
            },
        },
    }

    _finalize_object_summaries(results)

    assert results["objects"]["cube"]["success_rate"] == 0.75
    assert results["objects"]["can"]["success_rate"] == 0.0
    assert results["summary"]["success_rate"] == 0.5

    log_path = tmp_path / "eval.log"
    save_eval_log(str(log_path), results)
    log_text = log_path.read_text()
    assert "Overall success rate: 0.5000" in log_text
    assert "cube: success_rate=0.7500" in log_text
    assert "can: success_rate=0.0000" in log_text
