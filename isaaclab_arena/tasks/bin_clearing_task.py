# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bin clearing task: pick N objects from a source bin and place them all into a destination bin."""

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.bin_clearing_rate import BinClearingRateMetric
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import objects_on_destinations
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


@configclass
class BinClearingTerminationsCfg:
    """Termination terms for the bin clearing task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@register_task
class BinClearingTask(TaskBase):
    """Bin clearing task: pick all objects from a source bin and place them into a destination.

    Success (early termination) fires when ALL objects are on the destination with low velocity.
    Otherwise the episode runs until timeout. Object drops do NOT terminate the episode.

    Metrics:
      - success_rate: fraction of episodes where ALL objects were placed (strict)
      - bin_clearing_rate: average fraction of objects placed per episode (partial credit)
    """

    def __init__(
        self,
        pick_up_object_list: list[Asset],
        destination_location: Asset,
        background_scene: Asset,
        episode_length_s: float | None = None,
        force_threshold: float = 1.0,
        velocity_threshold: float = 0.5,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert len(pick_up_object_list) > 0, "At least one object is required"

        self.pick_up_object_list = pick_up_object_list
        self.destination_location = destination_location
        self.background_scene = background_scene
        self.force_threshold = force_threshold
        self.velocity_threshold = velocity_threshold

        self.contact_sensor_name_list = []
        self.pick_up_object_contact_sensor_list = []
        for obj in pick_up_object_list:
            sensor_cfg = obj.get_contact_sensor_cfg(contact_against_object=destination_location)
            sensor_name = f"contact_sensor_{obj.name}"
            self.pick_up_object_contact_sensor_list.append(sensor_cfg)
            self.contact_sensor_name_list.append(sensor_name)

        self.scene_config = self._make_scene_cfg()
        self.termination_cfg = self._make_termination_cfg()
        self.task_description = (
            f"Pick up all objects and place them into the {destination_location.name}"
            if task_description is None
            else task_description
        )

    def _make_scene_cfg(self):
        fields: list[tuple[str, type, ContactSensorCfg]] = []
        for sensor_name, sensor_cfg in zip(
            self.contact_sensor_name_list, self.pick_up_object_contact_sensor_list
        ):
            fields.append((sensor_name, type(sensor_cfg), sensor_cfg))
        SceneCfg = make_configclass("SceneCfg", fields)
        return SceneCfg()

    def _make_termination_cfg(self):
        object_cfg_list = [SceneEntityCfg(obj.name) for obj in self.pick_up_object_list]
        contact_sensor_cfg_list = [SceneEntityCfg(name) for name in self.contact_sensor_name_list]

        success = TerminationTermCfg(
            func=objects_on_destinations,
            params={
                "object_cfg_list": object_cfg_list,
                "contact_sensor_cfg_list": contact_sensor_cfg_list,
                "force_threshold": self.force_threshold,
                "velocity_threshold": self.velocity_threshold,
            },
        )
        return BinClearingTerminationsCfg(success=success)

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return None

    def get_task_description(self) -> str:
        return self.task_description

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("Mimic data generation is not supported for BinClearingTask")

    def get_metrics(self) -> list[MetricBase]:
        object_names = [obj.name for obj in self.pick_up_object_list]
        return [
            SuccessRateMetric(),
            BinClearingRateMetric(
                object_names=object_names,
                contact_sensor_names=self.contact_sensor_name_list,
                force_threshold=self.force_threshold,
                velocity_threshold=self.velocity_threshold,
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object_list[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )
