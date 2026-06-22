"""Verify Pi0PiperAdapter format: run one env reset, extract obs, pack request, test unpack."""
import sys

sys.argv = [
    "verify_adapter.py",
    "--headless",
    "--enable_cameras",
    "--policy_type", "zero_action",
    "--num_steps", "2",
    "--num_envs", "1",
    "pick_and_place_piper",
    "--pick_up_object", "rubiks_cube_hot3d_robolab",
    "--hdr", "home_office_robolab",
]

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)

import numpy as np
import torch
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

# Build full parser
full_parser = get_isaaclab_arena_cli_parser()
add_policy_runner_arguments(full_parser)
full_parser = get_isaaclab_arena_environments_cli_parser(full_parser)
args_cli = full_parser.parse_args(sys.argv[1:])

arena_builder = get_arena_builder_from_cli(args_cli)
env, cfg = arena_builder.make_registered_and_return_cfg()
obs, _ = env.reset()

# Warmup: run enough steps for textures/RTX to stabilize before capturing images
zero_action = torch.zeros(1, env.action_space.shape[-1], device="cuda")
warmup_steps = 30
for _ in range(warmup_steps):
    obs, _, _, _, _ = env.step(zero_action)

print("\n" + "=" * 60)
print("OBSERVATION STRUCTURE")
print("=" * 60)
for group_key, group_val in obs.items():
    print(f"\n[{group_key}]")
    if isinstance(group_val, dict):
        for k, v in group_val.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if v.numel() < 20:
                    print(f"    values: {v[0].cpu().numpy()}")
            else:
                print(f"  {k}: {type(v)}")
    elif hasattr(group_val, 'shape'):
        print(f"  shape={group_val.shape}, dtype={group_val.dtype}")

print("\n" + "=" * 60)
print("ADAPTER EXTRACT + PACK TEST")
print("=" * 60)

from isaaclab_arena_openpi.policy.piper_adapter import Pi0PiperAdapter
adapter = Pi0PiperAdapter()

try:
    extracted = adapter.extract(obs, env_id=0)

    # Save first-frame camera images
    from PIL import Image
    import os
    save_dir = "/workspaces/isaaclab_arena/docs/picture"
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(extracted.cam_top_image).save(os.path.join(save_dir, "cam_top.png"))
    Image.fromarray(extracted.cam_left_wrist_image).save(os.path.join(save_dir, "cam_left_wrist.png"))
    Image.fromarray(extracted.cam_right_wrist_image).save(os.path.join(save_dir, "cam_right_wrist.png"))
    print(f"\nSaved first-frame images to {save_dir}/")

    print(f"\nExtracted successfully:")
    print(f"  cam_top_image: shape={extracted.cam_top_image.shape}, dtype={extracted.cam_top_image.dtype}")
    print(f"  cam_left_wrist_image: shape={extracted.cam_left_wrist_image.shape}")
    print(f"  cam_right_wrist_image: shape={extracted.cam_right_wrist_image.shape}")
    print(f"  left_joint_position: shape={extracted.left_joint_position.shape}, values={extracted.left_joint_position}")
    print(f"  right_joint_position: shape={extracted.right_joint_position.shape}, values={extracted.right_joint_position}")
    print(f"  left_gripper_position: {extracted.left_gripper_position}")
    print(f"  right_gripper_position: {extracted.right_gripper_position}")

    request = adapter.pack_request(extracted, "Pick up the cube.")
    print(f"\nPacked request:")
    print(f"  keys: {list(request.keys())}")
    print(f"  state: shape={request['state'].shape}, values={request['state']}")
    print(f"  gripper_position: shape={request['gripper_position'].shape}, values={request['gripper_position']}")
    print(f"  images keys: {list(request['images'].keys())}")
    for img_key, img_val in request['images'].items():
        print(f"    {img_key}: shape={img_val.shape}, dtype={img_val.dtype}, range=[{img_val.min()}, {img_val.max()}]")
    print(f"  prompt: {request['prompt']}")

    # Test unpack_actions with simulated server response
    print(f"\n--- Simulated unpack_actions ---")
    fake_actions = np.random.randn(15, 14).astype(np.float32)
    fake_actions[:, 6] = 0.8    # left gripper: 0.8 > 0.5 = open
    fake_actions[:, 13] = 0.2   # right gripper: 0.2 <= 0.5 = closed
    result = adapter.unpack_actions(fake_actions)
    print(f"  Server output shape: {fake_actions.shape}")
    print(f"  Arena input shape:   {result.shape}")
    print(f"  Arena action[0]: {result[0]}")
    print(f"  Expected: [left_arm(6), right_arm(6), left_grip_cmd, right_grip_cmd]")
    print(f"  Gripper commands: left={result[0, 12]} (expect 1.0=open), right={result[0, 13]} (expect 0.0=close)")

    # Verify action dim matches env
    env_action_dim = env.action_space.shape[-1]
    print(f"\n  Env action_space dim: {env_action_dim}")
    print(f"  Adapter output dim:   {result.shape[1]}")
    print(f"  Match: {'YES' if env_action_dim == result.shape[1] else 'NO - MISMATCH!'}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

env.close()
app_launcher.app.close()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
