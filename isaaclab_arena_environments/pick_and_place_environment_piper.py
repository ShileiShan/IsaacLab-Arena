# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class PickAndPlaceEnvironmentPiper(ExampleEnvironmentBase):

    name: str = "pick_and_place_piper"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On, RandomAroundSolution, RotateAroundSolution
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        import math

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.pick_up_object)(
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=1.0,
                    linear_damping=2.0,
                    angular_damping=2.0,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.00),
            }
        )

        # Step 2: Table reference as anchor
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        # Step 3: Two material boxes — front (pick, near robot) and back (place, far from robot)
        # Box dimensions: 0.6m(y) × 0.4m(x) × 0.1m(z). Centers separated by 0.45m to avoid overlap.
        # Each box must have a unique instance_name so Scene stores them under different keys.
        _box_collision_props = {
            "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.001),
        }

        pick_box = self.asset_registry.get_asset_by_name(args_cli.pick_box)(
            instance_name="pick_box", spawn_cfg_addon=_box_collision_props
        )
        pick_box.add_relation(IsAnchor())
        pick_box.set_initial_pose(Pose(position_xyz=(0.328, 0.0, 0.08), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        # pick_box.add_relation(On(table_reference))
        # pick_box.add_relation(AtPosition(x = 0.45,y=0.0))
        # pick_box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 2))

        place_box = self.asset_registry.get_asset_by_name(args_cli.place_box)(
            instance_name="place_box", spawn_cfg_addon=_box_collision_props
        )
        place_box.add_relation(IsAnchor())
        place_box.set_initial_pose(Pose(position_xyz=(0.72, 0.0, 0.08), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        # place_box.add_relation(On(table_reference))
        # place_box.add_relation(AtPosition(x=0.9, y=0.0))
        # place_box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 2))

        # Step 4: Pick object sits above the pick box
        # pick_up_object.add_relation(AtPosition(x=0.4, y=0.0, z=0.86))
        # pick_up_object.add_relation(AtPosition(x=0.25, y=0.0, z=0.86))
        pick_up_object.add_relation(On(pick_box))
        pick_up_object.add_relation(RandomAroundSolution(x_half_m=0.05, y_half_m=0.1, yaw_half_rad=1.0))

        # Step 5: Additional objects on table (optional CLI arg)
        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in args_cli.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Step 6: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 7: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            initial_pose=Pose(position_xyz=(0.03, 0.0, 0.05)),
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Step 8: Compose the scene
        scene = Scene(
            assets=[
                background,
                light,
                pick_box,
                place_box,
                pick_up_object,
                table_reference,
                *additional_table_objects,
            ]
        )

        # Step 9: Define the task — place the pick_up_object into the place_box
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=place_box,
            background_scene=background,
            episode_length_s=20.0,
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            env_cfg.sim.dt = 1 / 300
            # Render every 6 physics steps → ~50Hz camera rate, avoids OgnSdOnNewFrame overrun
            env_cfg.sim.render_interval = 6
            from isaaclab_physx.physics import PhysxCfg
            if env_cfg.sim.physics is None:
                env_cfg.sim.physics = PhysxCfg()
            # Note: enable_ccd is not supported with GPU dynamics (PhysX limitation).
            env_cfg.sim.physics.solve_articulation_contact_last = True

            # Appended after events_cfg is fully assembled (including placement_reset), so
            # this runs last within mode="reset" and always sees the object already placed.
            from isaaclab.managers import EventTermCfg
            from isaaclab_arena.tasks.events import wait_for_objects_to_settle
            setattr(
                env_cfg.events,
                "wait_for_objects_to_settle",
                EventTermCfg(
                    func=wait_for_objects_to_settle,
                    mode="reset",
                    params={
                        "object_names": [pick_up_object.name],
                        "velocity_threshold": args_cli.settle_velocity_threshold,
                        "max_steps": args_cli.settle_max_steps,
                    },
                ),
            )
            return env_cfg

        # Step 10: Assemble the environment
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
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=1500.0)
        parser.add_argument("--pick_up_object", type=str, default="rubiks_cube_hot3d_robolab")
        parser.add_argument("--pick_box", type=str, default="material_box_003_kinematic",
                            help="Asset name for the pick box (near robot, holds the pick object)")
        parser.add_argument("--place_box", type=str, default="material_box_003_kinematic",
                            help="Asset name for the place box (far from robot, target destination)")
        parser.add_argument(
            "--additional_table_objects",
            nargs="*",
            type=str,
            default=[],
            help="Extra objects to place on the table alongside the pick-up object",
        )
        parser.add_argument(
            "--robot_x_offset",
            type=float,
            default=0.1,
            help="Robot base X position (meters). Increase to move robot closer to the boxes.",
        )
        parser.add_argument(
            "--settle_velocity_threshold",
            type=float,
            default=0.05,
            help="Max object linear/angular speed (m/s or rad/s) to consider objects settled after reset",
        )
        parser.add_argument(
            "--settle_max_steps",
            type=int,
            default=150,
            help="Max physics steps to wait for objects to settle after reset before giving up",
        )
