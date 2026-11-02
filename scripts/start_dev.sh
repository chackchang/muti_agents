#!/usr/bin/env bash
# 一键启动本地开发环境：Docker 依赖 → API → ingestion worker。
#
# 用法：
#   bash scripts/start_dev.sh          # 启动；Ctrl+C 一并停止 API 与 worker
#   bash scripts/start_dev.sh status   # 查看两个进程是否在跑 + 健康检查
#   bash scripts/start_dev.sh stop     # 停止（终端被关掉后也能用）
#
# 可选环境变量：
#   API_PORT=8001                      # API 端口，默认 8000
#   STARTUP_TIMEOUT=900                # 等 API 就绪的秒数（含 GE-M3 预热），默认 600
#
# 日志：logs/api.log、logs/worker.log；pid：logs/api.pid、logs/worker.pid
# Docker 依赖脚本只管启动，不会自动 docker compose down。

set -uo pipefail

# 统一用 UTF-8 写 stdout/stderr：Windows 默认 cp936，日志里会出现乱码字节
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
HOST="${HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
LOG_DIR="$ROOT_DIR/logs"
API_LOG="$LOG_DIR/api.log"
WORKER_LOG="$LOG_DIR/worker.log"
API_PID=""
WORKER_PID=""

log() { printf '[start_dev] %s\n' "$*"; }
die() { printf '[start_dev] ERROR: %s\n' "$*" >&2; exit 1; }

# 结束进程；Windows 上连同它拉起的子进程一起收掉（worker 会启动 ingest 子进程）。
stop_process() {
    local pid="$1" name="$2"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    log "Stopping $name (pid $pid)..."

    local winpid=""
    if command -v ps >/dev/null 2>&1; then
        winpid="$(ps -W 2>/dev/null | awk -v p="$pid" '$1 == p {print $4; exit}')"
    fi
    if [[ -n "$winpid" ]] && command -v taskkill >/dev/null 2>&1; then
        taskkill //PID "$winpid" //T //F >/dev/null 2>&1 || true
    fi

    # 兜底：进程树没收干净就按 pid 再杀一次
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 0.5
        kill -9 "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    trap - INT TERM EXIT
    stop_process "$WORKER_PID" worker
    stop_process "$API_PID" api
    rm -f "$LOG_DIR/api.pid" "$LOG_DIR/worker.pid"
    log "Stopped. Docker services are still running (use 'docker compose down' to stop them)."
}

# 轮询日志等某个进程就绪；进程中途退出或超时都直接失败并打印日志尾部。
wait_for_log() {
    local file="$1" pattern="$2" pid="$3" name="$4" timeout="$5"
    local i
    for ((i = 0; i < timeout; i++)); do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 20 "$file" >&2
            die "$name exited before becoming ready"
        fi
        sleep 1
    done
    tail -n 20 "$file" >&2
    die "$name did not become ready within ${timeout}s"
}

check_port_free() {
    "$PYTHON" - "$HOST" "$API_PORT" <<'PY' || die "port $HOST:$API_PORT is already in use"
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket()
try:
    sock.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

ensure_docker() {
    command -v docker >/dev/null 2>&1 || die "docker not found; install Docker Desktop first"
    log "Ensuring docker services (postgres / qdrant / rabbitmq)..."
    docker compose up -d >/dev/null 2>&1 || die "docker compose up failed; is Docker Desktop running?"

    local i
    for ((i = 0; i < 60; i++)); do
        local pg_ok=1 qdrant_ok=1 rabbit_ok=1
        docker compose exec -T postgres pg_isready -U postgres -d postgres >/dev/null 2>&1 || pg_ok=0
        curl -sf -m 2 "http://127.0.0.1:6333/readyz" >/dev/null 2>&1 || qdrant_ok=0
        docker compose exec -T rabbitmq rabbitmq-diagnostics -q ping >/dev/null 2>&1 || rabbit_ok=0
        if [[ $pg_ok -eq 1 && $qdrant_ok -eq 1 && $rabbit_ok -eq 1 ]]; then
            return 0
        fi
        sleep 1
    done
    die "docker services did not become ready within 60s (docker compose ps)"
}

running_pid() {
    local pidfile="$1" pid=""
    [[ -f "$pidfile" ]] && pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        printf '%s' "$pid"
    fi
}

do_start() {
    [[ -x "$PYTHON" ]] || die ".venv not found at $PYTHON (run 'uv sync' first)"
    mkdir -p "$LOG_DIR" out

    local existing
    existing="$(running_pid "$LOG_DIR/api.pid")$(running_pid "$LOG_DIR/worker.pid")"
    [[ -z "$existing" ]] || die "already running; use 'bash scripts/start_dev.sh stop' first"

    check_port_free
    ensure_docker

    : >"$API_LOG"
    log "Starting API on http://$HOST:$API_PORT (log: logs/api.log)"
    "$PYTHON" -m uvicorn agent.api:app --host "$HOST" --port "$API_PORT" >"$API_LOG" 2>&1 &
    API_PID=$!
    printf '%s' "$API_PID" >"$LOG_DIR/api.pid"
    trap cleanup INT TERM EXIT

    log "Waiting for API startup (GE-M3 warmup, up to ${STARTUP_TIMEOUT}s)..."
    wait_for_log "$API_LOG" "Application startup complete" "$API_PID" "API" "$STARTUP_TIMEOUT"

    : >"$WORKER_LOG"
    log "Starting ingestion worker (log: logs/worker.log)"
    "$PYTHON" -u -m agent.ingestion.ingestion_worker >"$WORKER_LOG" 2>&1 &
    WORKER_PID=$!
    printf '%s' "$WORKER_PID" >"$LOG_DIR/worker.pid"

    wait_for_log "$WORKER_LOG" "consuming queue" "$WORKER_PID" "worker" 120

    cat <<EOF

[start_dev] Ready — Ctrl+C stops API and worker (docker services stay up)
    API     http://$HOST:$API_PORT   pid=$API_PID   log=logs/api.log
    Worker  consuming ingestion_tasks   pid=$WORKER_PID   log=logs/worker.log

  在另一个终端手动测试：
    curl -s http://$HOST:$API_PORT/health
    curl -s -X POST http://$HOST:$API_PORT/search/hybrid -H "Content-Type: application/json" \\
         -d '{"query":"What is ISBNet?","limit":3}' > out/search.json
    curl -s -X POST http://$HOST:$API_PORT/chat -H "Content-Type: application/json" \\
         -d '{"message":"What is ISBNet?","search_type":"hybrid"}' > out/chat.json

  状态：bash scripts/start_dev.sh status
  停止：Ctrl+C，或 bash scripts/start_dev.sh stop
EOF

    # 任一进程退出就收掉另一个，避免留下半个环境
    wait -n "$API_PID" "$WORKER_PID"
    log "One process exited; shutting down the rest."
    exit 1
}

do_stop() {
    local name pidfile pid stopped=0
    for name in api worker; do
        pidfile="$LOG_DIR/$name.pid"
        pid="$(running_pid "$pidfile")"
        if [[ -n "$pid" ]]; then
            stop_process "$pid" "$name"
            stopped=1
        fi
        rm -f "$pidfile"
    done
    if [[ $stopped -eq 1 ]]; then
        log "Stopped."
    else
        log "Nothing running."
    fi
}

do_status() {
    local name pidfile pid
    for name in api worker; do
        pidfile="$LOG_DIR/$name.pid"
        pid="$(running_pid "$pidfile")"
        if [[ -n "$pid" ]]; then
            log "$name: running (pid $pid)"
        else
            log "$name: not running"
        fi
    done
    if command -v curl >/dev/null 2>&1; then
        printf '[start_dev] health: '
        curl -s -m 3 "http://$HOST:$API_PORT/health" || printf 'API not reachable'
        echo
    fi
}

case "${1:-start}" in
    start) do_start ;;
    stop) do_stop ;;
    status) do_status ;;
    *)
        echo "usage: bash scripts/start_dev.sh [start|stop|status]" >&2
        exit 2
        ;;
esac
