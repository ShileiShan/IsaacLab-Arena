# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-pinocchio DLS IK solver for a single Piper arm (no casadi required).

Joints 1-3 and 5-6 are controllable; joint4, joint7, joint8 are locked in
a reduced model so the solver operates on a 5-DOF chain.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import pinocchio as pin

    _PIN_AVAILABLE = True
except ImportError:
    pin = None
    _PIN_AVAILABLE = False

# Path to bundled URDF (copied from LW-BenchHub)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_DEFAULT_URDF = os.path.join(_ASSETS_DIR, "piper_description", "urdf", "piper_description.urdf")
_DEFAULT_PACKAGE_DIRS = [_ASSETS_DIR]


class PiperDLSIK:
    """Damped Least Squares IK for one Piper arm using pinocchio.

    The reduced model locks joint4 / joint7 / joint8, leaving 5 DOF
    (joint1, joint2, joint3, joint5, joint6).  The EE frame is
    ``gripper_base`` as present in the bundled URDF.

    Safety mechanisms (matching LW-BenchHub):
    - Error gating: if ||err_6D|| > err_gate_thresh use previous q
    - Per-step delta limiting: clip Δq to ±max_delta_per_step
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        package_dirs: list[str] | None = None,
        dls_lambda: float = 2e-2,
        max_iters: int = 100,
        tol: float = 1e-4,
        step_gain: float = 1.0,
        err_gate_thresh: float = 0.6,
        max_delta_per_step: float = 0.4,
        locked_joint4_value: float = 0.0,
    ):
        assert _PIN_AVAILABLE, "pinocchio is required for PiperDLSIK"

        if urdf_path is None:
            urdf_path = _DEFAULT_URDF
        if package_dirs is None:
            package_dirs = _DEFAULT_PACKAGE_DIRS

        full_model = pin.buildModelFromUrdf(urdf_path)

        # Build lock configuration: joint4 at user-specified value (must match
        # the actual sim state since the IK reduced model assumes this is fixed),
        # joint7/joint8 (gripper fingers) at 0.
        lock_q = pin.neutral(full_model)
        lock_ids = []
        for jname, lock_val in (
            ("joint4", float(locked_joint4_value)),
            ("joint7", 0.0),
            ("joint8", 0.0),
        ):
            jid = full_model.getJointId(jname)
            if 0 < jid < full_model.njoints:
                idx_q = full_model.joints[jid].idx_q
                lock_q[idx_q] = lock_val
                lock_ids.append(jid)

        self._model = pin.buildReducedModel(full_model, lock_ids, lock_q)
        self._data = self._model.createData()

        # Locate EE frame
        self._frame_id = self._model.getFrameId("gripper_base")
        assert self._frame_id < len(self._model.frames), "gripper_base frame not found"

        self._lambda2 = float(dls_lambda) ** 2
        self._max_iters = int(max_iters)
        self._tol = float(tol)
        self._gain = float(step_gain)
        self._err_gate_thresh = float(err_gate_thresh)
        self._max_delta = float(max_delta_per_step)

        self._q_lower = np.array(self._model.lowerPositionLimit)
        self._q_upper = np.array(self._model.upperPositionLimit)
        self._nq = self._model.nq

        # Per-env warm-start cache; initialised lazily
        self._last_q: np.ndarray | None = None  # (num_envs, nq)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        return self._nq

    def reset(self, env_ids: np.ndarray | None = None):
        """Clear warm-start cache (all envs or a subset)."""
        if self._last_q is None:
            return
        if env_ids is None:
            self._last_q[:] = 0.0
        else:
            self._last_q[env_ids] = 0.0

    def solve(
        self,
        targets: np.ndarray,
        warm_q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve IK for a batch of EE targets.

        Args:
            targets: (B, 7) array, each row [px, py, pz, qw, qx, qy, qz].
            warm_q:  optional (B, nq) starting configuration; uses internal
                     cache when None.

        Returns:
            q_out: (B, nq) joint positions in reduced-model order
                   [joint1, joint2, joint3, joint5, joint6].
        """
        B = int(targets.shape[0])
        self._ensure_cache(B)

        if warm_q is None:
            warm_q = self._last_q.copy()

        q_out = np.empty((B, self._nq), dtype=np.float64)
        for i in range(B):
            q_out[i] = self._solve_single(targets[i], warm_q[i])

        self._last_q = q_out.copy()
        return q_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_cache(self, batch_size: int):
        if self._last_q is None or self._last_q.shape[0] != batch_size:
            self._last_q = np.zeros((batch_size, self._nq), dtype=np.float64)

    def _target_se3(self, pose_wxyz: np.ndarray) -> "pin.SE3":
        pos = pose_wxyz[:3].astype(np.float64)
        w, x, y, z = pose_wxyz[3:7].astype(np.float64)
        R = pin.Quaternion(w, x, y, z).normalized().toRotationMatrix()
        return pin.SE3(R, pos)

    def _solve_single(self, target_wxyz: np.ndarray, q0: np.ndarray) -> np.ndarray:
        target = self._target_se3(target_wxyz)
        q = np.clip(q0.astype(np.float64), self._q_lower, self._q_upper)

        for _ in range(self._max_iters):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            oMf = self._data.oMf[self._frame_id]
            err = pin.log6(oMf.inverse() * target).vector  # (6,)
            if np.linalg.norm(err) < self._tol:
                break

            J = pin.computeFrameJacobian(
                self._model, self._data, q, self._frame_id, pin.LOCAL
            )  # (6, nq)

            # DLS: Δq = J^T (J J^T + λ² I)^{-1} err
            JJT = J @ J.T + self._lambda2 * np.eye(6)
            dq = self._gain * J.T @ np.linalg.solve(JJT, err)

            q = np.clip(q + dq, self._q_lower, self._q_upper)

        # --- error gating (position only) ---
        # Gate on Euclidean position distance, not log6 norm (which mixes meters and
        # radians and would block large-rotation targets even when position is reachable).
        prev_q = q0
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._frame_id]
        pos_err = np.linalg.norm(oMf.translation - target.translation)

        if pos_err > self._err_gate_thresh:
            q = prev_q.astype(np.float64)

        # --- per-step delta limiting ---
        delta = q - prev_q
        delta = np.clip(delta, -self._max_delta, self._max_delta)
        q = prev_q + delta
        return np.clip(q, self._q_lower, self._q_upper)
