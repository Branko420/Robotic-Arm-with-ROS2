#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/workspace/logs/robot_agents"
mkdir -p "$LOG_DIR"
cd /workspace

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[ERROR] ros2 is not loaded. Open Terminator normally or run:"
  echo "source /workspace/robot_env.sh"
  exit 1
fi

if ! ros2 pkg prefix robot_system >/dev/null 2>&1; then
  echo "[ERROR] robot_system is not loaded. Run:"
  echo "source /workspace/robot_env.sh"
  exit 1
fi

redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 127.0.0.1 --port 6379

pkill -f "ros2 run robot_system workspace_scan_agent" 2>/dev/null || true
pkill -f "ros2 run robot_system planning_agent" 2>/dev/null || true
pkill -f "ros2 run robot_system realtime_agent" 2>/dev/null || true

sleep 0.2

start_node() {
  local name="$1"
  shift
  echo "[START] $name"
  nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
}

start_node workspace_scan_agent ros2 run robot_system workspace_scan_agent
start_node planning_agent ros2 run robot_system planning_agent

if [[ "${START_REALTIME:-1}" == "1" ]]; then
  start_node realtime_agent ros2 run robot_system realtime_agent
fi

echo "✅ Started"
echo
echo "Running:"
echo "  workspace_scan_agent  = tool backend: AI bbox + depth -> workspace points"
echo "  planning_agent = plans to existing points"
echo "  realtime_agent        = voice/tools"
echo
echo "RViz topics:"
echo "  /digital_twin/debug_ai_input_image    Image"
echo "  /digital_twin/motion_point_markers    MarkerArray"
echo "  /digital_twin/planner_markers         MarkerArray"
echo
echo "Logs:"
echo "  tail -f $LOG_DIR/workspace_scan_agent.log"
echo "  tail -f $LOG_DIR/planning_agent.log"
echo "  tail -f $LOG_DIR/realtime_agent.log"
