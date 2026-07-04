#!/usr/bin/env bash
# Start (or stop) all three ablation-arm MCP servers at once so VS Code can
# switch arms with a single toggle in the MCP panel:
#   ct  -> http://localhost:8000/mcp  (all tools)
#   kg  -> http://localhost:8001/mcp  (7 reasoning/validation tools)
#   rag -> http://localhost:8002/mcp  (query_hcm only)
# The `base` arm = no server (toggle all three off in VS Code).
#
# Usage:  ./run_ablation_arms.sh start | stop | status
set -euo pipefail
cd "$(dirname "$0")"

PIDDIR=".ablation_pids"
mkdir -p "$PIDDIR"

start() {
  for arm in ct:mcp_server_fastapi.py kg:mcp_server_kg_only.py rag:mcp_server_rag_only.py; do
    name="${arm%%:*}"; script="${arm##*:}"
    pidfile="$PIDDIR/$name.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "[$name] already running (pid $(cat "$pidfile"))"; continue
    fi
    nohup uv run python "$script" >"$PIDDIR/$name.log" 2>&1 &
    echo $! >"$pidfile"
    echo "[$name] started (pid $!) -> log $PIDDIR/$name.log"
  done
  echo "Give them ~10s to load, then check VS Code MCP panel for tool counts (kg=7, rag=1)."
}

stop() {
  for pidfile in "$PIDDIR"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "[$name] stopped (pid $pid)"; fi
    rm -f "$pidfile"
  done
}

status() {
  for name in ct kg rag; do
    pidfile="$PIDDIR/$name.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "[$name] UP (pid $(cat "$pidfile"))"
    else
      echo "[$name] down"
    fi
  done
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "Usage: $0 start|stop|status" >&2; exit 1 ;;
esac
