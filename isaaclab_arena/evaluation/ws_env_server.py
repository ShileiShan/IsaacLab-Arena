# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""WebSocket environment server for Isaac Sim.

Exposes an Isaac Lab Arena environment as a WebSocket service using the
msgpack-numpy protocol defined in ``infer_debug/isaac_sim_ws_server_interface.md``.

External clients (e.g. OpenPI) connect and issue ``reset`` / ``step`` / ``stop``
commands.  The server returns observations (joint positions, gripper states,
camera images) and accepts joint-position actions.

Usage (inside the Docker container):
    # Single GPU
    CUDA_VISIBLE_DEVICES=1 python isaaclab_arena/evaluation/ws_env_server.py \\
        --livestream 2 --viz kit \\
        --enable_cameras --num_envs 1 \\
        --ws_port 8765 \\
        pick_and_place_piper \\
        --pick_up_object banana_ycb_robolab \\
        --hdr home_office_robolab

    # Multi-GPU (2 GPUs, ports 8765 and 8766)
    python -m torch.distributed.run --nnode=1 --nproc_per_node=2 \\
        isaaclab_arena/evaluation/ws_env_server.py \\
        --distributed --enable_cameras --num_envs 10 \\
        --ws_port 8765 \\
        pick_and_place_piper \\
        --pick_up_object banana_ycb_robolab \\
        --hdr home_office_robolab
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

GRIPPER_MAX_OPENING = 0.035  # meters


def _add_ws_server_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("WebSocket Server")
    group.add_argument("--ws_port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    group.add_argument("--ws_host", type=str, default="0.0.0.0", help="WebSocket bind address (default: 0.0.0.0)")
    group.add_argument(
        "--warm_up_steps", type=int, default=5,
        help="Steps to hold position after reset for renderer warm-up (default: 5)",
    )
    group.add_argument(
        "--export_usd", type=str, default=None,
        help="Export the simulation stage to a USD file after env creation",
    )
    group.add_argument(
        "--language_instruction", type=str, default=None,
        help="Language instruction for the task (unused by server, logged for reference)",
    )
    return parser


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor, warp array, or numpy array to numpy."""
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        return x.numpy()
    return np.asarray(x)


def _warm_up_renderer(env, obs: dict, num_steps: int = 5) -> dict:
    """Hold-position steps to let the ray-tracing renderer produce valid frames."""
    policy_obs = obs.get("policy", {})
    left_pos = policy_obs.get("left_joint_pos")
    right_pos = policy_obs.get("right_joint_pos")

    if left_pos is None or right_pos is None:
        action_dim = env.unwrapped.action_manager.total_action_dim
        hold_action = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
    else:
        num_envs = left_pos.shape[0]
        hold_action = torch.cat([
            left_pos,
            right_pos,
            torch.zeros(num_envs, 1, device=left_pos.device),
            torch.zeros(num_envs, 1, device=left_pos.device),
        ], dim=1)

    for _ in range(num_steps):
        obs, _, _, _, _ = env.step(hold_action)
    return obs


def _extract_obs(obs: dict, num_envs: int) -> dict:
    """Convert internal observation dict to the interface format.

    Single-env (num_envs==1): strips the batch dimension.
    Multi-env: keeps the N-batch dimension.
    """
    policy = obs.get("policy", {})
    camera = obs.get("camera_obs", {})

    left_joint = _to_numpy(policy["left_joint_pos"])
    right_joint = _to_numpy(policy["right_joint_pos"])
    left_grip = _to_numpy(policy["left_gripper_pos"]) * GRIPPER_MAX_OPENING
    right_grip = _to_numpy(policy["right_gripper_pos"]) * GRIPPER_MAX_OPENING

    result = {
        "left_joint_pos": left_joint.astype(np.float32),
        "right_joint_pos": right_joint.astype(np.float32),
        "left_gripper_pos": left_grip.astype(np.float32),
        "right_gripper_pos": right_grip.astype(np.float32),
    }

    for cam_key in ("first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb"):
        img = camera.get(cam_key)
        if img is not None:
            img_np = _to_numpy(img)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            result[cam_key] = img_np

    if num_envs == 1:
        result = {k: v[0] for k, v in result.items()}

    return result


def _build_action_tensor(action_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert interface 14D action(s) to internal action tensor.

    Interface layout: [left_arm(6), right_arm(6), left_grip_m(1), right_grip_m(1)]
    Internal layout:  [left_arm(6), right_arm(6), left_grip_m(1), right_grip_m(1)]

    The SymmetricGripperPositionActionCfg with max_opening=0.035 accepts g in [0, 0.035]
    directly, so no conversion is needed — the interface and internal formats match.
    """
    action_np = np.asarray(action_np, dtype=np.float32)
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]
    return torch.from_numpy(action_np).to(device)


def _serve(env, num_envs: int, device: torch.device, warm_up_steps: int, host: str, port: int):
    """Run a WebSocket server with the main thread driving the sim loop.

    Architecture:
    - WebSocket server runs in a daemon thread, accepting connections.
    - When a request arrives, the WS handler puts it into a queue and blocks
      waiting for a response.
    - The main thread polls the request queue, processes env.reset()/env.step(),
      puts the response back, and yields to Isaac Sim's internal loop via
      SimulationApp.update() so rendering/livestream keep working.
    """
    import queue
    import threading
    import time

    import msgpack
    from websockets.sync.server import serve as ws_serve

    def _encode_numpy(obj):
        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": str(obj.dtype),
                b"shape": list(obj.shape),
            }
        return obj

    def _decode_numpy(obj):
        if b"__ndarray__" in obj:
            return np.frombuffer(obj[b"data"], dtype=obj[b"dtype"]).reshape(obj[b"shape"])
        return obj

    def pack(data) -> bytes:
        return msgpack.packb(data, default=_encode_numpy, use_bin_type=True)

    def unpack(raw: bytes):
        return msgpack.unpackb(raw, object_hook=_decode_numpy, raw=False)

    req_queue: queue.Queue = queue.Queue()
    resp_queues: dict[int, queue.Queue] = {}
    _lock = threading.Lock()
    _client_counter = [0]

    def handle(websocket):
        client = websocket.remote_address
        with _lock:
            _client_counter[0] += 1
            client_id = _client_counter[0]
            resp_queues[client_id] = queue.Queue()
        print(f"[WS Server] Client connected: {client} (id={client_id})")

        try:
            for raw in websocket:
                req = unpack(raw)
                cmd = req.get("cmd")

                if cmd == "stop":
                    print(f"[WS Server] Client {client} sent stop")
                    break

                req_queue.put((client_id, req))
                response = resp_queues[client_id].get()
                websocket.send(pack(response))
                with _t_obs_sent_lock:
                    _t_obs_sent[client_id] = time.time()

        except Exception as e:
            print(f"[WS Server] Connection error: {e}")
        finally:
            with _lock:
                resp_queues.pop(client_id, None)
            with _t_obs_sent_lock:
                _t_obs_sent.pop(client_id, None)
            print(f"[WS Server] Client disconnected: {client}")

    server = ws_serve(handle, host, port)
    ws_thread = threading.Thread(target=server.serve_forever, daemon=True)
    ws_thread.start()

    print(f"[WS Server] WebSocket server listening on {host}:{port}")
    print(f"[WS Server] num_envs={num_envs}, warm_up_steps={warm_up_steps}")

    import omni.kit.app
    sim_app = omni.kit.app.get_app()
    step_counter = [0]
    _idle_ticks = [0]
    _t_step_start = [None]   # wall time of first step
    _t_last_print = [None]   # wall time of last freq print
    _ep_start_times = [None] * num_envs   # wall time when each env's current episode started
    _ep_durations: list[list[float]] = [[] for _ in range(num_envs)]  # completed episode durations per env
    _ep_durations_all: list[float] = []  # all completed episode durations across all envs
    # Per-step timing: sim vs inference+transfer
    _t_obs_sent: dict[int, float] = {}   # client_id -> wall time when obs was last sent
    _t_obs_sent_lock = threading.Lock()
    _sim_time_acc = [0.0]     # accumulated env.step() wall time
    _obs_extract_time_acc = [0.0]  # accumulated obs extraction (GPU->CPU copy) wall time
    _infer_time_acc = [0.0]   # accumulated inference+transfer wall time
    _timing_count = [0]       # steps with valid both-side measurements
    _t_serve_start = time.time()  # wall time when serving started

    while True:
        try:
            client_id, req = req_queue.get(block=False)
            _idle_ticks[0] = 0
        except queue.Empty:
            # Only call sim_app.update() when idle so we don't double-tick physics
            sim_app.update()
            _idle_ticks[0] += 1
            if _idle_ticks[0] % 500 == 0:
                print(f"[WS Server] Waiting for client... ({_idle_ticks[0]} idle ticks)", flush=True)
            time.sleep(0.001)
            continue

        cmd = req.get("cmd")

        if cmd == "reset":
            print(f"[WS Server] Processing reset...", flush=True)
            obs, _ = env.reset()
            if warm_up_steps > 0:
                obs = _warm_up_renderer(env, obs, warm_up_steps)
            response = _extract_obs(obs, num_envs)
            obs_keys = list(response.keys())
            obs_shapes = {k: v.shape if hasattr(v, "shape") else type(v).__name__ for k, v in response.items()}
            print(f"[WS Server] Reset done. obs_keys={obs_keys}", flush=True)
            print(f"[WS Server]   shapes: {obs_shapes}", flush=True)
            step_counter[0] = 0
            _t_step_start[0] = None
            _t_last_print[0] = None
            _now_reset = time.time()
            for _i in range(num_envs):
                _ep_start_times[_i] = _now_reset

        elif cmd == "step":
            action_raw = req.get("actions") if num_envs > 1 else req.get("action")
            if action_raw is None:
                action_raw = req.get("actions") or req.get("action")
            if action_raw is None:
                response = {"error": "step command missing 'action' or 'actions'"}
                print(f"[WS Server] Step ERROR: no action in request. keys={list(req.keys())}", flush=True)
            else:
                action_np = np.asarray(action_raw)
                action_tensor = _build_action_tensor(action_raw, device)
                if action_tensor.shape[0] != num_envs:
                    print(
                        f"[WS Server] ACTION SHAPE MISMATCH: received {action_tensor.shape},"
                        f" expected ({num_envs}, {action_tensor.shape[-1] if action_tensor.ndim > 1 else '?'})."
                        f" client_id={client_id}  req_keys={list(req.keys())}",
                        flush=True,
                    )
                _t_step_begin = time.time()
                with _t_obs_sent_lock:
                    _t_prev_obs_sent = _t_obs_sent.get(client_id)
                obs, reward, terminated, truncated, info = env.step(action_tensor)
                _t_step_end = time.time()
                _sim_time_acc[0] += _t_step_end - _t_step_begin
                if _t_prev_obs_sent is not None:
                    _infer_time_acc[0] += _t_step_begin - _t_prev_obs_sent
                    _timing_count[0] += 1

                _t_obs_extract_begin = time.time()
                response = _extract_obs(obs, num_envs)
                _obs_extract_time_acc[0] += time.time() - _t_obs_extract_begin
                done = _to_numpy(terminated | truncated)
                terminated_np = _to_numpy(terminated)
                truncated_np = _to_numpy(truncated)

                if num_envs == 1:
                    response["done"] = bool(done[0])
                    response["info"] = {}
                    if terminated_np[0] or truncated_np[0]:
                        term_mgr = env.unwrapped.termination_manager
                        fired = [n for n in term_mgr.active_terms
                                 if bool(_to_numpy(term_mgr.get_term(n))[0])]
                        _t_done = time.time()
                        if _ep_start_times[0] is not None:
                            _dur = _t_done - _ep_start_times[0]
                            _ep_durations[0].append(_dur)
                            _ep_durations_all.append(_dur)
                            _avg = sum(_ep_durations[0]) / len(_ep_durations[0])
                            _avg_all = sum(_ep_durations_all) / len(_ep_durations_all)
                        else:
                            _dur = float("nan")
                            _avg = float("nan")
                            _avg_all = float("nan")
                        _ep_start_times[0] = _t_done
                        _ep_tag = f"  [{_dur:.1f}s this ep, avg {_avg:.1f}s over {len(_ep_durations[0])} eps | global avg {_avg_all:.1f}s over {len(_ep_durations_all)} eps]"
                        if "success" in fired:
                            print(f"[WS Server] *** SUCCESS at step #{step_counter[0]+1} ***{_ep_tag}", flush=True)
                        elif truncated_np[0]:
                            print(f"[WS Server] --- TIMEOUT at step #{step_counter[0]+1} ---{_ep_tag}", flush=True)
                        else:
                            print(f"[WS Server] --- TERMINATED at step #{step_counter[0]+1}: {fired} ---{_ep_tag}", flush=True)
                else:
                    response["done"] = done.astype(bool)
                    response["reward"] = _to_numpy(reward).astype(np.float32)
                    response["infos"] = [{} for _ in range(num_envs)]
                    term_mgr = env.unwrapped.termination_manager
                    for i in range(num_envs):
                        if terminated_np[i] or truncated_np[i]:
                            fired = [n for n in term_mgr.active_terms
                                     if bool(_to_numpy(term_mgr.get_term(n))[i])]
                            _t_done = time.time()
                            if _ep_start_times[i] is not None:
                                _dur = _t_done - _ep_start_times[i]
                                _ep_durations[i].append(_dur)
                                _ep_durations_all.append(_dur)
                                _avg = sum(_ep_durations[i]) / len(_ep_durations[i])
                                _avg_all = sum(_ep_durations_all) / len(_ep_durations_all)
                            else:
                                _dur = float("nan")
                                _avg = float("nan")
                                _avg_all = float("nan")
                            _ep_start_times[i] = _t_done
                            _ep_tag = f"  [{_dur:.1f}s this ep, avg {_avg:.1f}s over {len(_ep_durations[i])} eps | global avg {_avg_all:.1f}s over {len(_ep_durations_all)} eps]"
                            if "success" in fired:
                                print(f"[WS Server] *** SUCCESS env_{i} at step #{step_counter[0]+1} ***{_ep_tag}", flush=True)
                            elif truncated_np[i]:
                                print(f"[WS Server] --- TIMEOUT env_{i} at step #{step_counter[0]+1} ---{_ep_tag}", flush=True)
                            else:
                                print(f"[WS Server] --- TERMINATED env_{i} at step #{step_counter[0]+1}: {fired} ---{_ep_tag}", flush=True)

                step_counter[0] += 1
                now = time.time()
                if _t_step_start[0] is None:
                    _t_step_start[0] = now
                    _t_last_print[0] = now
                elapsed_total = now - _t_step_start[0]
                elapsed_since_print = now - _t_last_print[0]
                if step_counter[0] <= 3 or elapsed_since_print >= 5.0:
                    avg_hz = step_counter[0] / elapsed_total if elapsed_total > 0 else 0.0
                    print(
                        f"[WS Server] Step #{step_counter[0]}: action_shape={action_np.shape}"
                        f"  done={response.get('done')}"
                        f"  wall_throughput={avg_hz:.1f} step/s (sim_ctrl=50Hz)",
                        flush=True,
                    )
                    if _timing_count[0] > 0:
                        avg_sim = _sim_time_acc[0] / _timing_count[0]
                        avg_obs = _obs_extract_time_acc[0] / _timing_count[0]
                        avg_infer = _infer_time_acc[0] / _timing_count[0]
                        total = avg_sim + avg_obs + avg_infer
                        sim_pct = avg_sim / total * 100 if total > 0 else 0.0
                        obs_pct = avg_obs / total * 100 if total > 0 else 0.0
                        infer_pct = avg_infer / total * 100 if total > 0 else 0.0
                        print(
                            f"[WS Server] Timing breakdown (avg over {_timing_count[0]} steps):"
                            f"  sim/render={avg_sim*1000:.1f}ms ({sim_pct:.0f}%)"
                            f"  obs_extract={avg_obs*1000:.1f}ms ({obs_pct:.0f}%)"
                            f"  transfer+infer={avg_infer*1000:.1f}ms ({infer_pct:.0f}%)",
                            flush=True,
                        )
                    total_eps = len(_ep_durations_all)
                    elapsed_min = (now - _t_serve_start) / 60.0
                    if total_eps > 0 and elapsed_min > 0:
                        eps_per_min = total_eps / elapsed_min
                        print(
                            f"[WS Server] Throughput: {total_eps} episodes in {elapsed_min:.1f}min"
                            f"  = {eps_per_min:.2f} eps/min",
                            flush=True,
                        )
                    _t_last_print[0] = now

        else:
            response = {"error": f"Unknown command: {cmd}"}

        with _lock:
            rq = resp_queues.get(client_id)
        if rq is not None:
            rq.put(response)


def main():
    """Run an Isaac Lab Arena environment as a WebSocket server.
    Use --distributed with torchrun for one process per GPU; each rank listens on ws_port + local_rank.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    if hasattr(args_cli, "distributed") and args_cli.distributed and world_size > 1:
        args_cli.device = f"cuda:{local_rank}"
        print(f"[WS Server] [Rank {local_rank}/{world_size}] Distributed mode: cuda:{local_rank}")

    with SimulationAppContext(args_cli):
        _add_ws_server_arguments(args_parser)
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        if hasattr(args_cli, "distributed") and args_cli.distributed and world_size > 1:
            args_cli.device = f"cuda:{local_rank}"
            args_cli.ws_port = args_cli.ws_port + local_rank
            print(f"[WS Server] [Rank {local_rank}/{world_size}] Port offset: ws_port={args_cli.ws_port}")

        arena_builder = get_arena_builder_from_cli(args_cli)
        env, cfg = arena_builder.make_registered_and_return_cfg(render_mode=None)

        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device

        if args_cli.export_usd:
            from isaaclab_arena.evaluation.policy_runner import _export_stage_usd
            _export_stage_usd(args_cli.export_usd)

        if args_cli.language_instruction:
            print(f"[WS Server] Language instruction: {args_cli.language_instruction}")

        print(f"[WS Server] Environment ready: {env.unwrapped.__class__.__name__}")
        print(f"[WS Server] Action dim: {env.unwrapped.action_manager.total_action_dim}")
        print(f"[WS Server] Device: {device}")

        _serve(
            env=env,
            num_envs=num_envs,
            device=device,
            warm_up_steps=args_cli.warm_up_steps,
            host=args_cli.ws_host,
            port=args_cli.ws_port,
        )


if __name__ == "__main__":
    main()
