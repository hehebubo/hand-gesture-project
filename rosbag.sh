#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./rosbag.sh record [topic...]
  ./rosbag.sh play <bag_path>
  ./rosbag.sh info <bag_path>

Defaults:
  record -> /hand_target /ik_joint_states /joint_command
  output -> bags/hand_session_YYYYMMDD_HHMMSS
EOF
}

MODE="${1:-record}"
shift || true

OUT_DIR="${OUT_DIR:-bags}"
DEFAULT_TOPICS=("/hand_target" "/ik_joint_states" "/joint_command")

case "$MODE" in
  record)
    mkdir -p "$OUT_DIR"
    TS="$(date +%Y%m%d_%H%M%S)"
    BAG_NAME="${BAG_NAME:-hand_session_${TS}}"
    BAG_PATH="${OUT_DIR}/${BAG_NAME}"
    if [ "$#" -gt 0 ]; then
      TOPICS=("$@")
    else
      TOPICS=("${DEFAULT_TOPICS[@]}")
    fi
    echo "Recording to: ${BAG_PATH}"
    echo "Topics: ${TOPICS[*]}"
    exec ros2 bag record "${TOPICS[@]}" -o "$BAG_PATH"
    ;;
  play)
    if [ "$#" -lt 1 ]; then
      echo "Missing bag path."
      usage
      exit 1
    fi
    exec ros2 bag play "$1"
    ;;
  info)
    if [ "$#" -lt 1 ]; then
      echo "Missing bag path."
      usage
      exit 1
    fi
    exec ros2 bag info "$1"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac
