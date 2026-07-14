# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven sequential evaluation for Double Piper single-object pick-and-place.

The runner expands a YAML config into one job per object/initial-position pair,
runs them sequentially, and reports per-object success rates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from collections import OrderedDict
from copy import deepcopy
from statistics import mean
from typing import Any

from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status


DEFAULT_CONFIG_PATH = "isaaclab_arena_environments/eval_jobs_configs/double_piper_single_object_eval.yaml"
DEFAULT_ENVIRONMENT = "pick_and_place_piper"
DEFAULT_EMBODIMENT = "double_piper_abs_joint_pos"
SUCCESS_RATE_KEY = "success_rate"
BOOLEAN_OPTIONAL_ARENA_ARGS = {"resolve_on_reset", "use_tiled_camera"}


def load_yaml_config(path: str) -> dict[str, Any]:
    import yaml

    assert os.path.exists(path), f"Evaluation config file does not exist: {path}"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    assert isinstance(config, dict), f"Evaluation config must be a YAML mapping, got {type(config).__name__}"
    return config


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("_") or "unnamed"


def _as_float_list(value: Any, *, expected_len: int, field_name: str) -> list[float]:
    assert isinstance(value, (list, tuple)), f"{field_name} must be a list/tuple, got {type(value).__name__}"
    assert len(value) == expected_len, f"{field_name} must contain {expected_len} values, got {len(value)}"
    return [float(item) for item in value]


def _get_policy_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    policy_cfg = config.get("policy", {})
    assert isinstance(policy_cfg, dict), "'policy' must be a mapping when provided"

    policy_type = config.get("policy_type", policy_cfg.get("type"))
    assert policy_type is not None, "Set either top-level 'policy_type' or 'policy.type'"

    policy_config_dict = deepcopy(config.get("policy_config_dict", {}))
    policy_config_dict.update(deepcopy(policy_cfg.get("config", {})))
    assert isinstance(policy_config_dict, dict), "policy_config_dict/policy.config must be mappings"
    return str(policy_type), policy_config_dict


def _get_base_arena_env_args(config: dict[str, Any]) -> dict[str, Any]:
    base_args = {
        "environment": DEFAULT_ENVIRONMENT,
        "embodiment": DEFAULT_EMBODIMENT,
    }
    base_args.update(deepcopy(config.get("arena_env_args", {})))
    for key in ("environment", "embodiment", "num_envs", "enable_cameras", "placement_seed", "env_spacing"):
        if key in config:
            base_args[key] = config[key]
    return base_args


def _normalize_object_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry}
    assert isinstance(entry, dict), f"Each object entry must be a string or mapping, got {type(entry).__name__}"
    object_name = entry.get("name", entry.get("object", entry.get("pick_up_object")))
    assert object_name is not None, "Object entries require 'name', 'object', or 'pick_up_object'"
    normalized = deepcopy(entry)
    normalized["name"] = str(object_name)
    return normalized


def _normalize_position_entry(entry: Any, index: int) -> dict[str, Any]:
    if isinstance(entry, (list, tuple)):
        return {"name": f"pos_{index:02d}", "bin_offset_xyz": list(entry)}
    assert isinstance(entry, dict), f"Position entries must be lists or mappings, got {type(entry).__name__}"
    normalized = deepcopy(entry)
    normalized.setdefault("name", f"pos_{index:02d}")
    return normalized


def _position_env_args(position: dict[str, Any]) -> dict[str, Any]:
    offset = position.get("bin_offset_xyz", position.get("offset_xyz", position.get("xyz")))
    assert offset is not None, "Each position requires 'bin_offset_xyz' (or alias 'offset_xyz'/'xyz')"
    env_args = {
        "pick_object_bin_offset_xyz": _as_float_list(
            offset,
            expected_len=3,
            field_name="position.bin_offset_xyz",
        )
    }
    if "rotation_xyzw" in position:
        env_args["pick_object_rotation_xyzw"] = _as_float_list(
            position["rotation_xyzw"],
            expected_len=4,
            field_name="position.rotation_xyzw",
        )
    if "arena_env_args" in position:
        assert isinstance(position["arena_env_args"], dict), "position.arena_env_args must be a mapping"
        env_args.update(deepcopy(position["arena_env_args"]))
    return env_args


def _encode_boolean_optional_args(arena_env_args: dict[str, Any]) -> dict[str, Any]:
    """Encode false values for argparse.BooleanOptionalAction as --no-<flag>."""
    encoded_args = deepcopy(arena_env_args)
    for key in BOOLEAN_OPTIONAL_ARENA_ARGS:
        if encoded_args.get(key) is False:
            encoded_args.pop(key)
            encoded_args[f"no-{key}"] = True
    return encoded_args


def build_jobs_from_config(config: dict[str, Any]) -> tuple[list[Job], dict[str, dict[str, Any]]]:
    """Expand a Double Piper object-position config into sequential jobs."""
    policy_type, policy_config_dict = _get_policy_config(config)
    base_arena_env_args = _get_base_arena_env_args(config)
    default_positions = config.get("positions")
    objects = config.get("objects")
    assert isinstance(objects, list) and objects, "Config requires a non-empty 'objects' list"

    jobs: list[Job] = []
    metadata_by_job_name: dict[str, dict[str, Any]] = {}
    used_names: set[str] = set()

    for object_index, raw_object in enumerate(objects):
        object_cfg = _normalize_object_entry(raw_object)
        object_name = object_cfg["name"]
        object_positions = object_cfg.get("positions", default_positions)
        assert isinstance(object_positions, list) and object_positions, (
            f"Object '{object_name}' requires a non-empty 'positions' list, either on the object or at top level"
        )

        object_arena_env_args = deepcopy(object_cfg.get("arena_env_args", {}))
        assert isinstance(object_arena_env_args, dict), f"Object '{object_name}' arena_env_args must be a mapping"

        for position_index, raw_position in enumerate(object_positions):
            position_cfg = _normalize_position_entry(raw_position, position_index)
            position_name = str(position_cfg["name"])

            arena_env_args = deepcopy(base_arena_env_args)
            arena_env_args.update(object_arena_env_args)
            arena_env_args["pick_up_object"] = object_name
            arena_env_args.update(_position_env_args(position_cfg))
            arena_env_args = _encode_boolean_optional_args(arena_env_args)

            num_envs = int(arena_env_args.get("num_envs", 1))
            num_steps = position_cfg.get("num_steps", object_cfg.get("num_steps", config.get("num_steps")))
            num_episodes = position_cfg.get(
                "num_episodes",
                object_cfg.get("num_episodes", config.get("num_episodes")),
            )
            assert not (
                num_steps is not None and num_episodes is not None
            ), f"Job for object '{object_name}' position '{position_name}' has both num_steps and num_episodes"

            job_base_name = _sanitize_name(f"{object_index:02d}_{object_name}_{position_index:02d}_{position_name}")
            job_name = job_base_name
            duplicate_index = 1
            while job_name in used_names:
                duplicate_index += 1
                job_name = f"{job_base_name}_{duplicate_index}"
            used_names.add(job_name)

            job_policy_config = deepcopy(policy_config_dict)
            job_policy_config.update(deepcopy(object_cfg.get("policy_config_dict", {})))
            job_policy_config.update(deepcopy(position_cfg.get("policy_config_dict", {})))

            language_instruction = position_cfg.get(
                "language_instruction",
                object_cfg.get("language_instruction", config.get("language_instruction")),
            )

            jobs.append(
                Job(
                    name=job_name,
                    num_envs=num_envs,
                    arena_env_args=Job.convert_args_dict_to_cli_args_list(arena_env_args),
                    policy_type=policy_type,
                    num_steps=num_steps,
                    num_episodes=num_episodes,
                    policy_config_dict=job_policy_config,
                    language_instruction=language_instruction,
                )
            )
            metadata_by_job_name[job_name] = {
                "object": object_name,
                "position": position_name,
                "bin_offset_xyz": arena_env_args["pick_object_bin_offset_xyz"],
                "rotation_xyzw": arena_env_args.get("pick_object_rotation_xyzw", [0.0, 0.0, 0.0, 1.0]),
            }

    return jobs, metadata_by_job_name


def config_requests_cameras(config: dict[str, Any], jobs: list[Job]) -> bool:
    if bool(config.get("enable_cameras", False)):
        return True
    app_cfg = config.get("app", {})
    if isinstance(app_cfg, dict) and bool(app_cfg.get("enable_cameras", False)):
        return True
    for job in jobs:
        if "--enable_cameras" in job.arena_env_args:
            return True
    return False


def _apply_app_config(args_cli: argparse.Namespace, config: dict[str, Any]) -> None:
    app_cfg = config.get("app", {})
    assert isinstance(app_cfg, dict), "'app' must be a mapping when provided"
    for key, value in app_cfg.items():
        setattr(args_cli, key, value)


def _runner_option(args_cli: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    runner_cfg = config.get("runner", {})
    assert isinstance(runner_cfg, dict), "'runner' must be a mapping when provided"
    cli_value = getattr(args_cli, name, None)
    if isinstance(cli_value, bool):
        return bool(cli_value or runner_cfg.get(name, default))
    if name in runner_cfg:
        return runner_cfg[name]
    return cli_value if cli_value is not None else default


def _run_job(job: Job, *, video: bool, video_dir: str):
    from gymnasium.wrappers import RecordVideo

    from isaaclab_arena.evaluation.eval_runner import get_policy_from_job, load_env
    from isaaclab_arena.evaluation.policy_runner import rollout_policy

    env = None
    policy = None
    try:
        render_mode = "rgb_array" if video else None
        env = load_env(job.arena_env_args, job.name, render_mode=render_mode)
        policy = get_policy_from_job(job)

        if job.num_steps is None and job.num_episodes is None:
            if policy.has_length():
                job.num_steps = policy.length()
            else:
                raise AssertionError(f"Job '{job.name}' requires num_steps or num_episodes")

        if video:
            if job.num_steps is not None:
                video_length = job.num_steps
            else:
                video_length = job.num_episodes * env.unwrapped.max_episode_length
            job_video_dir = os.path.join(video_dir, job.name)
            print(f"[INFO] Recording video for job '{job.name}' -> {job_video_dir}")
            env = RecordVideo(
                env,
                video_folder=job_video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )

        return rollout_policy(
            env,
            policy,
            num_steps=job.num_steps,
            num_episodes=job.num_episodes,
            language_instruction=job.language_instruction,
        )
    finally:
        from isaaclab_arena.evaluation.eval_runner import _close_job_resources

        _close_job_resources(policy, env)


def _append_job_result(
    results: dict[str, Any],
    job: Job,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    status: Status,
) -> None:
    from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types

    plain_metrics = metrics_to_plain_python_types(metrics or {})
    job_result = {
        **metadata,
        "status": status.value,
        "num_envs": job.num_envs,
        "num_steps": job.num_steps,
        "num_episodes": job.num_episodes,
        "metrics": plain_metrics,
    }
    results["jobs"][job.name] = job_result

    object_name = metadata["object"]
    object_result = results["objects"].setdefault(
        object_name,
        {
            "positions": [],
            "success_rate": None,
            "num_positions": 0,
            "num_completed_positions": 0,
            "num_failed_positions": 0,
        },
    )
    object_result["positions"].append({"job": job.name, **job_result})


def _finalize_object_summaries(results: dict[str, Any]) -> None:
    all_rates = []
    num_positions = 0
    num_completed_positions = 0
    num_failed_positions = 0

    for object_result in results["objects"].values():
        rates = [
            float(position["metrics"][SUCCESS_RATE_KEY])
            for position in object_result["positions"]
            if position["status"] == Status.COMPLETED.value and SUCCESS_RATE_KEY in position["metrics"]
        ]
        object_result["num_positions"] = len(object_result["positions"])
        object_result["num_completed_positions"] = sum(
            position["status"] == Status.COMPLETED.value for position in object_result["positions"]
        )
        object_result["num_failed_positions"] = sum(
            position["status"] == Status.FAILED.value for position in object_result["positions"]
        )
        object_result["success_rate"] = float(mean(rates)) if rates else None

        all_rates.extend(rates)
        num_positions += object_result["num_positions"]
        num_completed_positions += object_result["num_completed_positions"]
        num_failed_positions += object_result["num_failed_positions"]

    results["summary"] = {
        "success_rate": float(mean(all_rates)) if all_rates else None,
        "num_positions": num_positions,
        "num_completed_positions": num_completed_positions,
        "num_failed_positions": num_failed_positions,
    }


def print_object_summary(results: dict[str, Any]) -> None:
    from prettytable import PrettyTable

    summary = results.get("summary", {})
    overall_success_rate = summary.get("success_rate")
    overall_success_rate_text = "n/a" if overall_success_rate is None else f"{overall_success_rate:.4f}"

    table = PrettyTable(field_names=["Object", "Success Rate", "Completed Positions", "Failed Positions"])
    for object_name, object_result in results["objects"].items():
        success_rate = object_result["success_rate"]
        success_rate_text = "n/a" if success_rate is None else f"{success_rate:.4f}"
        table.add_row([
            object_name,
            success_rate_text,
            f"{object_result['num_completed_positions']}/{object_result['num_positions']}",
            object_result["num_failed_positions"],
        ])
    print("\nDouble Piper Single-Object Evaluation Summary")
    print(f"Overall success rate: {overall_success_rate_text}")
    print(table)


def save_results(path: str, results: dict[str, Any]) -> None:
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved evaluation results to: {path}")


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def save_eval_log(path: str, results: dict[str, Any]) -> None:
    if not path:
        return

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summary = results.get("summary", {})
    lines = [
        "Double Piper Single-Object Evaluation",
        "=" * 45,
        f"Config: {results.get('config_path', '')}",
        f"Overall success rate: {_format_rate(summary.get('success_rate'))}",
        f"Completed positions: {summary.get('num_completed_positions', 0)}/{summary.get('num_positions', 0)}",
        f"Failed positions: {summary.get('num_failed_positions', 0)}",
        "",
        "Per-object success rates:",
    ]

    for object_name, object_result in results["objects"].items():
        lines.append(
            "- "
            f"{object_name}: success_rate={_format_rate(object_result.get('success_rate'))}, "
            f"completed={object_result.get('num_completed_positions', 0)}/"
            f"{object_result.get('num_positions', 0)}, "
            f"failed={object_result.get('num_failed_positions', 0)}"
        )

    lines += [
        "",
        "Per-position job results:",
    ]

    for job_name, job_result in results["jobs"].items():
        metrics = job_result.get("metrics", {})
        lines.append(
            "- "
            f"{job_name}: object={job_result.get('object')}, "
            f"position={job_result.get('position')}, "
            f"offset={job_result.get('bin_offset_xyz')}, "
            f"status={job_result.get('status')}, "
            f"success_rate={_format_rate(metrics.get(SUCCESS_RATE_KEY))}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Saved evaluation log to: {path}")


def add_double_piper_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval_config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Double Piper single-object YAML evaluation config.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos for each eval job.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/eval/videos/double_piper_single_object",
        help="Root directory for recorded videos. Each job gets a subdirectory.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue evaluation with remaining jobs when a job fails instead of stopping immediately.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="double_piper_single_object_eval_results.json",
        help="Path to write aggregate JSON results. Set to an empty string to disable.",
    )
    parser.add_argument(
        "--eval_log",
        type=str,
        default="eval.log",
        help="Path to write the human-readable evaluation summary. Set to an empty string to disable.",
    )


def main() -> None:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_parser = get_isaaclab_arena_cli_parser()
    add_double_piper_eval_arguments(args_parser)
    args_cli = args_parser.parse_args()

    config = load_yaml_config(args_cli.eval_config)
    _apply_app_config(args_cli, config)
    jobs, metadata_by_job_name = build_jobs_from_config(config)

    assert not getattr(args_cli, "distributed", False), "Distributed evaluation is not supported by this runner"
    if config_requests_cameras(config, jobs):
        args_cli.enable_cameras = True

    video = bool(_runner_option(args_cli, config, "video", False))
    video_dir = str(_runner_option(args_cli, config, "video_dir", args_cli.video_dir))
    continue_on_error = bool(_runner_option(args_cli, config, "continue_on_error", False))
    output_json = str(_runner_option(args_cli, config, "output_json", args_cli.output_json))
    eval_log = str(_runner_option(args_cli, config, "eval_log", args_cli.eval_log))

    results: dict[str, Any] = {
        "config_path": args_cli.eval_config,
        "summary": {},
        "jobs": OrderedDict(),
        "objects": OrderedDict(),
    }

    with SimulationAppContext(args_cli):
        job_manager = JobManager(jobs)
        job_manager.print_jobs_info()

        for job in job_manager:
            metadata = metadata_by_job_name[job.name]
            try:
                print(
                    f"[INFO] Evaluating object='{metadata['object']}' "
                    f"position='{metadata['position']}' offset={metadata['bin_offset_xyz']}"
                )
                metrics = _run_job(job, video=video, video_dir=video_dir)
                job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)
                _append_job_result(results, job, metadata, metrics or {}, Status.COMPLETED)
            except Exception as exc:
                job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                _append_job_result(results, job, metadata, {}, Status.FAILED)
                print(f"Job {job.name} failed with error: {exc}")
                print(f"Traceback: {traceback.format_exc()}")
                if not continue_on_error:
                    raise

        _finalize_object_summaries(results)
        job_manager.print_jobs_info()
        print_object_summary(results)
        save_results(output_json, results)
        save_eval_log(eval_log, results)


if __name__ == "__main__":
    main()
