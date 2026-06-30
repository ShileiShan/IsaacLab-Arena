# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bin clearing environment for double Piper: pick N objects from a source bin and place into destination."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class BinClearingEnvironmentPiper(ExampleEnvironmentBase):

    name: str = "bin_clearing_piper"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, RandomAroundSolution
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.bin_clearing_task import BinClearingTask
        from isaaclab_arena.utils.pose import Pose

        # Step 1: Background + ground plane (catches objects that fall off the table)
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, -0.3)),
            spawner_cfg=sim_utils.GroundPlaneCfg(visible=False),
        )

        # Step 2: Table reference as anchor
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        # Step 3: Pick box (source) and place box (destination)
        _box_collision_props = {"collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)}

        pick_box = self.asset_registry.get_asset_by_name(args_cli.pick_box)(
            instance_name="pick_box", spawn_cfg_addon=_box_collision_props
        )
        pick_box.add_relation(IsAnchor())
        pick_box.set_initial_pose(
            Pose(position_xyz=(0.328, 0.0, 0.08), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068))
        )

        place_box = self.asset_registry.get_asset_by_name(args_cli.place_box)(
            instance_name="place_box", spawn_cfg_addon=_box_collision_props
        )
        place_box.add_relation(IsAnchor())
        place_box.set_initial_pose(
            Pose(position_xyz=(0.75, 0.0, 0.08), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068))
        )

        # Step 4: Randomly select N objects from pool and place in pick box
        # Scale randomization range down when many objects to help solver find valid layouts.
        # z_randomization should be kept small (<=0.02) to avoid objects spawning too high
        # and falling out of the pick box.
        num_objects = args_cli.num_objects
        x_half = min(0.05, 0.15 / max(num_objects, 1))
        y_half = min(0.1, 0.30 / max(num_objects, 1))

        # Enable per-body higher solver iterations to reduce penetration through bin walls.
        _bin_object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=5.0,
        )

        selected_names = random.choices(args_cli.object_pool, k=num_objects)
        pick_up_objects = []
        for i, obj_name in enumerate(selected_names):
            obj = self.asset_registry.get_asset_by_name(obj_name)(
                instance_name=f"pick_object_{i}",
                spawn_cfg_addon={
                    "rigid_props": _bin_object_rigid_props,
                    "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                },
            )
            obj.add_relation(On(pick_box))
            obj.add_relation(
                RandomAroundSolution(
                    x_half_m=x_half,
                    y_half_m=y_half,
                    z_half_m=args_cli.z_randomization / 2,
                    z_offset_m=args_cli.z_randomization / 2,
                    yaw_half_rad=1.0,
                )
            )
            pick_up_objects.append(obj)

        # Step 5: Lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 6: Embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Step 7: Scene
        scene = Scene(
            assets=[
                background,
                ground_plane,
                light,
                pick_box,
                place_box,
                table_reference,
                *pick_up_objects,
            ]
        )

        # Step 8: Task
        episode_length_s = args_cli.num_objects * args_cli.time_per_object
        task = BinClearingTask(
            pick_up_object_list=pick_up_objects,
            destination_location=place_box,
            background_scene=background,
            episode_length_s=episode_length_s,
            force_threshold=args_cli.force_threshold,
            velocity_threshold=args_cli.velocity_threshold,
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            env_cfg.sim.dt = 1 / 300
            from isaaclab_physx.physics import PhysxCfg
            if env_cfg.sim.physics is None:
                env_cfg.sim.physics = PhysxCfg()
            # Note: enable_ccd is not supported with GPU dynamics (PhysX limitation).
            # Tunneling is mitigated via contact_offset on each object and higher solver iterations.
            env_cfg.sim.physics.solve_articulation_contact_last = True
            return env_cfg

        # Step 9: Assemble
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="double_piper_abs_joint_pos")
        parser.add_argument("--num_objects", type=int, default=3, help="Number of objects to place in pick box")
        parser.add_argument(
            "--time_per_object",
            type=float,
            default=20.0,
            help="Seconds allowed per object (episode_length = num_objects * time_per_object)",
        )
        parser.add_argument(
            "--object_pool",
            nargs="*",
            type=str,
            default=["rubiks_cube_hot3d_robolab"],
            help="Pool of object asset names to randomly select from",
        )
        parser.add_argument(
            "--z_randomization",
            type=float,
            default=0.02,
            help="Half-extent in Z for object spawn height randomization (meters)",
        )
        parser.add_argument("--pick_box", type=str, default="material_box_003_kinematic")
        parser.add_argument("--place_box", type=str, default="material_box_003_kinematic")
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--force_threshold", type=float, default=1.0, help="Contact force threshold for success")
        parser.add_argument("--velocity_threshold", type=float, default=0.5, help="Velocity threshold for success")
        parser.add_argument("--teleop_device", type=str, default=None)
