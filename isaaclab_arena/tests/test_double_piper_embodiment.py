# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check tests for the Double Piper embodiment."""

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
ENABLE_CAMERAS = True


# ---------------------------------------------------------------------------
# Test: joint names and shapes load correctly (no cameras)
# ---------------------------------------------------------------------------


def _test_double_piper_joints(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.double_piper.double_piper import DoublePiperAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    embodiment = DoublePiperAbsoluteJointPositionEmbodiment(enable_cameras=False)

    env_def = IsaacLabArenaEnvironment(
        name="double_piper_joints_test",
        embodiment=embodiment,
        scene=Scene(assets=[background]),
    )

    env = ArenaEnvBuilder(env_def, args_cli).make_registered()
    env.reset()

    robot = env.scene["robot"]
    print("\n=== 关节名 ===")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:02d}] {name}")

    print("\n=== 关节位置 shape ===")
    print(f"  joint_pos: {robot.data.joint_pos.shape}")

    print("\n=== 末端执行器 ===")
    ee = env.scene["ee_frame"]
    print(f"  target_pos_w shape: {ee.data.target_pos_w.shape}")
    print(f"  target names: {ee.target_frame_names}")

    print("\n=== 机器人根节点位置（检查是否固定）===")
    print(f"  root_pos_w: {robot.data.root_pos_w}")

    # 步进几步，确认动作空间 shape 正确
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for _ in range(3):
        obs, _, terminated, _, _ = env.step(actions)

    print("\n=== 动作空间 ===")
    print(f"  shape: {env.action_space.shape}")

    print("\n=== 观测空间 keys ===")
    for key in obs:
        print(f"  {key}")

    env.close()
    return True


def test_double_piper_joints():
    assert run_simulation_app_function(
        _test_double_piper_joints, headless=HEADLESS, enable_cameras=False
    )


# ---------------------------------------------------------------------------
# Test: cameras load and produce correct-shape RGB observations
# ---------------------------------------------------------------------------


def _test_double_piper_cameras(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.double_piper.double_piper import DoublePiperAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    embodiment = DoublePiperAbsoluteJointPositionEmbodiment(enable_cameras=True)

    env_def = IsaacLabArenaEnvironment(
        name="double_piper_cameras_test",
        embodiment=embodiment,
        scene=Scene(assets=[background]),
    )

    env = ArenaEnvBuilder(env_def, args_cli).make_registered()
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs, _, _, _, _ = env.step(actions)

    print("\n=== 相机观测 ===")
    camera_obs = obs.get("camera_obs", {})
    for cam_key, tensor in camera_obs.items():
        print(f"  {cam_key}: shape={tensor.shape}, dtype={tensor.dtype}")
        # 期望 shape: (num_envs, H, W, C) 或 (num_envs, C, H, W)
        assert tensor.shape[-1] == 3 or tensor.shape[1] == 3, (
            f"期望 RGB 3 通道, 实际 shape: {tensor.shape}"
        )

    assert len(camera_obs) > 0, "未找到任何相机观测，检查 enable_cameras 和 prim 路径"
    print(f"\n共加载 {len(camera_obs)} 个相机观测")

    env.close()
    return True


@pytest.mark.with_cameras
def test_double_piper_cameras():
    assert run_simulation_app_function(
        _test_double_piper_cameras, headless=HEADLESS, enable_cameras=ENABLE_CAMERAS
    )
