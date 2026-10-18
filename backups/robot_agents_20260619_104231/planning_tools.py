#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import redis
from agents import function_tool
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
PLANNING_REQUEST_STREAM = os.getenv(
    "PLANNING_REQUEST_STREAM",
    "stream:planning_requests",
)
PLANNING_RESULT_STREAM = os.getenv(
    "PLANNING_RESULT_STREAM",
    "stream:planning_results",
)
WORKSPACE_SCAN_REQUEST_STREAM = os.getenv(
    "WORKSPACE_SCAN_REQUEST_STREAM",
    "stream:workspace_scan_requests",
)
WORKSPACE_SCAN_RESULT_STREAM = os.getenv(
    "WORKSPACE_SCAN_RESULT_STREAM",
    "stream:workspace_scan_results",
)
MOTION_POINTS_KEY = os.getenv("MOTION_POINTS_KEY", "workspace:motion_points")
PLANNING_TOOL_TIMEOUT_SEC = float(
    os.getenv("PLANNING_TOOL_TIMEOUT_SEC", "18")
)
WORKSPACE_SCAN_TOOL_TIMEOUT_SEC = float(
    os.getenv("WORKSPACE_SCAN_TOOL_TIMEOUT_SEC", "45")
)

REDIS = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _stream_tail(stream: str) -> str:
    rows = REDIS.xrevrange(stream, count=1)
    return rows[0][0] if rows else "0-0"


def _wait_for_result(
    stream: str,
    request_id: str,
    after_id: str,
    timeout_sec: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_id = after_id

    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        rows = REDIS.xread(
            {stream: last_id},
            count=20,
            block=min(500, remaining_ms),
        )

        for _stream_name, entries in rows:
            for entry_id, fields in entries:
                last_id = entry_id
                try:
                    payload = json.loads(fields.get("payload", "{}"))
                except Exception:
                    continue
                if payload.get("request_id") == request_id:
                    return payload

    return {
        "ok": False,
        "request_id": request_id,
        "state": "TOOL_TIMEOUT",
        "error": f"No result from {stream} within {timeout_sec:.1f}s",
    }


def _request(
    request_stream: str,
    result_stream: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> Dict[str, Any]:
    request_id = payload.setdefault(
        "request_id",
        f"request-{uuid.uuid4().hex[:10]}",
    )
    result_tail = _stream_tail(result_stream)
    REDIS.xadd(
        request_stream,
        {"payload": json.dumps(payload, ensure_ascii=False)},
    )
    return _wait_for_result(
        result_stream,
        request_id,
        result_tail,
        timeout_sec,
    )


@function_tool
def scan_workspace_and_make_points() -> str:
    """Refresh the complete workspace inventory and create a point for every classified entity."""
    result = _request(
        WORKSPACE_SCAN_REQUEST_STREAM,
        WORKSPACE_SCAN_RESULT_STREAM,
        {
            "request_id": f"workspace-scan-{uuid.uuid4().hex[:10]}",
            "action": "scan_workspace_and_make_points",
        },
        WORKSPACE_SCAN_TOOL_TIMEOUT_SEC,
    )
    return json.dumps(result, ensure_ascii=False)


@function_tool
def list_workspace_points() -> str:
    """Return all currently known objects, flat placement surfaces, aliases and saved positions."""
    raw = REDIS.get(MOTION_POINTS_KEY)
    if not raw:
        return json.dumps(
            {"ok": False, "state": "NO_WORKSPACE_POINTS"},
            ensure_ascii=False,
        )

    try:
        document = json.loads(raw)
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "state": "INVALID_WORKSPACE_POINTS",
                "error": str(exc),
            },
            ensure_ascii=False,
        )

    points = []
    for point in document.get("motion_points", []):
        points.append(
            {
                "point": point.get("point"),
                "label": point.get("label"),
                "display_name_bg": point.get("display_name_bg"),
                "role": point.get("role"),
                "colors": point.get("colors", []),
                "aliases": point.get("aliases", []),
                "center": point.get("center"),
                "top_position": point.get("top_position"),
                "approach_position": point.get("approach_position"),
                "grasp_position": point.get("grasp_position"),
                "place_position": point.get("place_position"),
                "dimensions": point.get("dimensions"),
                "geometry_quality": point.get("geometry_quality"),
                "safety_blocker": point.get("safety_blocker", False),
            }
        )

    return json.dumps(
        {
            "ok": True,
            "count": len(points),
            "points": points,
        },
        ensure_ascii=False,
    )


@function_tool
def control_robot_in_rviz(
    action: str,
    target: str = "nearest_object",
    target_position: str = "center",
    execute: bool = True,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> str:
    """
    Plan or move the arm in RViz.

    Supported actions: preview, move, approach, grasp, pick, place, home, stop.
    target can be a saved alias or a natural description such as "зеления куб"
    or "белия лист". target_position can be center, top, approach, grasp or place.
    dx/dy/dz are additional base_link offsets in metres.
    """
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            "request_id": f"plan-{uuid.uuid4().hex[:10]}",
            "action": action,
            "target": target,
            "target_position": target_position,
            "execute": execute,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )
    return json.dumps(result, ensure_ascii=False)


@function_tool
def pick_and_place_in_rviz(
    source: str,
    destination: str,
    execute: bool = True,
) -> str:
    """
    Pick one visible object and place it on another visible destination.

    Examples:
    - source="зеления куб", destination="белия лист"
    - source="nearest_cube", destination="white_paper"
    """
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            "request_id": f"pick-place-{uuid.uuid4().hex[:10]}",
            "action": "pick_and_place",
            "source": source,
            "destination": destination,
            "execute": execute,
        },
        max(PLANNING_TOOL_TIMEOUT_SEC, 30.0),
    )
    return json.dumps(result, ensure_ascii=False)


@function_tool
def correct_last_robot_target(
    forward_m: float = 0.0,
    left_m: float = 0.0,
    up_m: float = 0.0,
    target: Optional[str] = None,
) -> str:
    """
    Persist a correction for the last or named target.

    Positive forward_m is +X, positive left_m is +Y and positive up_m is +Z.
    Use negative values for backward, right and down.
    """
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            "request_id": f"correct-{uuid.uuid4().hex[:10]}",
            "action": "correct",
            "target": target,
            "dx": forward_m,
            "dy": left_m,
            "dz": up_m,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )
    return json.dumps(result, ensure_ascii=False)
