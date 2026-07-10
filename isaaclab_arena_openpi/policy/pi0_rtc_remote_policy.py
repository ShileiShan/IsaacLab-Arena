# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena_openpi.policy.pi0_remote_config import DEFAULT_VARIANT, Pi0RemotePolicyArgs
from isaaclab_arena_openpi.policy.pi0_remote_policy import (
    Pi0EmbodimentAdapter,
    Pi0RemotePolicy,
    _resolve_openpi_embodiment_adapter,
)


@dataclass
class Pi0RTCRemotePolicyArgs(Pi0RemotePolicyArgs):
    """Connection + runtime config for ``Pi0RTCRemotePolicy``."""

    rtc_control_hz: float = 30.0


@register_policy
class Pi0RTCRemotePolicy(Pi0RemotePolicy):
    """openpi RTC remote policy.

    RTC servers return one action per ``infer()`` call instead of an open-loop
    action chunk. This policy is intended for closed-loop use with the double
    Piper ``pick_and_place_piper`` environment and ``Pi0PiperAdapter``.
    """

    name = "pi0_rtc_remote"
    config_class = Pi0RTCRemotePolicyArgs

    def __init__(self, config: Pi0RTCRemotePolicyArgs, openpi_embodiment_adapter: Pi0EmbodimentAdapter) -> None:
        super().__init__(config, openpi_embodiment_adapter=openpi_embodiment_adapter)
        rtc_control_hz = getattr(config, "rtc_control_hz", 0.0)
        self._control_dt_s = 1.0 / rtc_control_hz if rtc_control_hz > 0.0 else None
        self._last_control_start_s: float | None = None
        self._last_server_timing: dict[str, Any] | None = None

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Pi0 RTC Remote Policy",
            "Arguments for the openpi RTC remote client.",
        )
        group.add_argument(
            "--openpi_embodiment_adapter",
            type=str,
            default="piper",
            choices=["droid", "piper"],
            help="Openpi-side embodiment adapter for obs / action wire format (default: piper).",
        )
        group.add_argument(
            "--policy_variant",
            type=str,
            default=DEFAULT_VARIANT,
            help=(
                f"openpi checkpoint variant (default: {DEFAULT_VARIANT})."
                " Kept for config parity with pi0_remote."
            ),
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Torch device for action tensors (default: cuda).",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="openpi RTC server host.")
        group.add_argument("--remote_port", type=int, default=8001, help="openpi RTC server port.")
        group.add_argument(
            "--rtc_control_hz",
            type=float,
            default=30.0,
            help="Optional start-to-start RTC control rate limiter in Hz. Use 0 to disable throttling.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Pi0RTCRemotePolicy:
        openpi_embodiment_adapter = _resolve_openpi_embodiment_adapter(args.openpi_embodiment_adapter)
        return Pi0RTCRemotePolicy(
            Pi0RTCRemotePolicyArgs(
                policy_variant=args.policy_variant,
                policy_device=args.policy_device,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                rtc_control_hz=args.rtc_control_hz,
            ),
            openpi_embodiment_adapter=openpi_embodiment_adapter,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Pi0RTCRemotePolicy:
        config_dict = dict(config_dict)
        adapter_key = config_dict.pop("openpi_embodiment_adapter", "piper")
        openpi_embodiment_adapter = _resolve_openpi_embodiment_adapter(adapter_key)
        return cls(Pi0RTCRemotePolicyArgs(**config_dict), openpi_embodiment_adapter=openpi_embodiment_adapter)

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "Pi0RTCRemotePolicy requires a non-empty language instruction"
            " (set via --language_instruction or on the task definition)."
        )

        self._throttle_control_rate()

        actions = []
        for env_id in range(env.unwrapped.num_envs):
            actions.append(self._fetch_single_action(observation, env_id))

        batch = np.stack(actions)
        return torch.from_numpy(batch).to(dtype=torch.float32, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        _ = env_ids
        for attr in ("_prev_left_grip_last", "_prev_right_grip_last"):
            if hasattr(self._openpi_embodiment_adapter, attr):
                delattr(self._openpi_embodiment_adapter, attr)
        self._last_control_start_s = None

    def _throttle_control_rate(self) -> None:
        if self._control_dt_s is None:
            return
        now = time.perf_counter()
        if self._last_control_start_s is not None:
            sleep_s = self._control_dt_s - (now - self._last_control_start_s)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        self._last_control_start_s = time.perf_counter()

    def _fetch_single_action(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        extracted = self._openpi_embodiment_adapter.extract(observation, env_id)
        request = self._openpi_embodiment_adapter.pack_request(extracted, self.task_description)

        t0 = time.perf_counter()
        response = self._call_server_with_retry(request)
        self._server_time_total_s += time.perf_counter() - t0
        self._server_call_count += 1
        self._last_server_timing = response.get("server_timing")

        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        elif actions.ndim == 2:
            actions = actions[:1]
        else:
            raise AssertionError(
                f"Expected RTC actions of shape ({self._openpi_embodiment_adapter.action_dim},)"
                f" or (H, {self._openpi_embodiment_adapter.action_dim}); got {actions.shape}"
            )

        assert actions.shape[1] == self._openpi_embodiment_adapter.action_dim, (
            f"Expected RTC actions with dim {self._openpi_embodiment_adapter.action_dim}; got {actions.shape}"
        )

        if hasattr(self._openpi_embodiment_adapter, "unpack_actions"):
            actions = self._openpi_embodiment_adapter.unpack_actions(actions)

        return actions[0].astype(np.float32, copy=False)
