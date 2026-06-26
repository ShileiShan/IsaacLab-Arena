# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.tasks.terminations import object_on_destination


class BinClearingRateRecorder(RecorderTerm):
    """Records the fraction of objects placed on the destination at episode end.

    At each pre-reset, evaluates each object's contact sensor to determine how many
    objects are currently on the destination. Records placed_count / total_count.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.object_names = cfg.object_names
        self.contact_sensor_names = cfg.contact_sensor_names
        self.force_threshold = cfg.force_threshold
        self.velocity_threshold = cfg.velocity_threshold
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        num_objects = len(self.object_names)
        placed_count = torch.zeros(len(env_ids), device=self._env.device)

        for obj_name, sensor_name in zip(self.object_names, self.contact_sensor_names):
            on_dest = object_on_destination(
                env=self._env,
                object_cfg=SceneEntityCfg(obj_name),
                contact_sensor_cfg=SceneEntityCfg(sensor_name),
                force_threshold=self.force_threshold,
                velocity_threshold=self.velocity_threshold,
            )
            placed_count += on_dest[env_ids].float()

        clearing_rate = placed_count / num_objects
        return self.name, clearing_rate


@configclass
class BinClearingRateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = BinClearingRateRecorder
    name: str = "bin_clearing_rate"
    object_names: list[str] = []
    contact_sensor_names: list[str] = []
    force_threshold: float = 1.0
    velocity_threshold: float = 0.5


def compute_bin_clearing_rate(recorded_metric_data: list[np.ndarray]) -> float:
    """Computes the average bin clearing rate across all episodes.

    Args:
        recorded_metric_data: List of arrays, each containing per-env clearing rates.

    Returns:
        Average clearing rate across all episodes (0.0 to 1.0).
    """
    num_demos = len(recorded_metric_data)
    if num_demos == 0:
        return 0.0
    all_rates = np.concatenate(recorded_metric_data)
    return float(np.mean(all_rates))


class BinClearingRateMetric(MetricBase):
    """Computes the average fraction of objects successfully placed per episode."""

    name = "bin_clearing_rate"
    recorder_term_name = "bin_clearing_rate"

    def __init__(
        self,
        object_names: list[str],
        contact_sensor_names: list[str],
        force_threshold: float = 1.0,
        velocity_threshold: float = 0.5,
    ):
        self.object_names = object_names
        self.contact_sensor_names = contact_sensor_names
        self.force_threshold = force_threshold
        self.velocity_threshold = velocity_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return BinClearingRateRecorderCfg(
            name=self.recorder_term_name,
            object_names=self.object_names,
            contact_sensor_names=self.contact_sensor_names,
            force_threshold=self.force_threshold,
            velocity_threshold=self.velocity_threshold,
        )

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_bin_clearing_rate,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
