#!/usr/bin/env bash

# =================================================================
# Ray Cluster Launcher (Port-Safe Fast Version)
# -----------------------------------------------------------------
# 目的: 快速启动多节点 Ray 集群，自动解决端口占用问题
# 优化: SSH 复用、并行操作、快速轮询、批量检查
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
else
  echo "[ERROR] 环境配置文件未找到: ${ENV_FILE}" >&2
  exit 1
fi

# --- 1. 默认配置与常量 ---
PROJECT_DIR="${SCRIPT_DIR}"
NPUS_PER_NODE=8
WAIT_TIME="${RAY_WAIT_TIME:-0.5}"
NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
PARALLEL_JOBS="${RAY_PARALLEL_JOBS:-8}"
FAST_MODE="${RAY_FAST_MODE:-false}"
PORT_SEARCH_RANGE="${RAY_PORT_RANGE:-20}"

# 端口配置
BASE_MASTER_PORT="${RAY_PORT:-6379}"
BASE_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
BASE_METRICS_PORT="${RAY_METRICS_PORT:-50000}"

# Worker 端口范围（避开 10000-20000 内部组件范围）
MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-30000}"
MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-35000}"

# SSH 连接复用配置
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
if [[ "${RAY_SSH_MUX:-true}" == "true" ]]; then
    SSH_OPTS="${SSH_OPTS} -o ControlMaster=auto -o ControlPersist=60s -o ControlPath=/tmp/ssh_mux_%h_%p_%r"
fi

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- 2. 帮助信息与日志函数 ---
usage() {
    cat <<EOF
Usage: $0 [options]

Starts a multi-node Ray cluster with automatic port conflict resolution.

Options:
  --project-dir DIR       Project directory containing 'set_env.sh' (default: $PROJECT_DIR)
  --node-list FILE        File containing list of nodes (default: $NODE_LIST_FILE)
  --port PORT             Base Ray master port (default: $BASE_MASTER_PORT)
  --dashboard-port PORT   Base Dashboard port (default: $BASE_DASHBOARD_PORT)
  --metrics-port PORT     Base Metrics port (default: $BASE_METRICS_PORT)
  --worker-port-start NUM Worker port range start (default: $MIN_WORKER_PORT)
  --worker-port-end NUM   Worker port range end (default: $MAX_WORKER_PORT)
  --npus-per-node NUM     Number of NPUs per node (default: $NPUS_PER_NODE)
  --wait-time SEC         Wait time for head node initialization (default: $WAIT_TIME)
  --force-cleanup         Force cleanup of existing Ray processes before starting
  --auto-port             Automatically find available ports (recommended)
  --fast-mode             Skip non-critical checks for faster startup
  --parallel-jobs NUM     Number of parallel jobs (default: $PARALLEL_JOBS)
  --help                  Show this help message

Environment Variables:
  RAY_PORT              - Base port for Ray head node
  RAY_DASHBOARD_PORT    - Base port for Ray dashboard
  RAY_FAST_MODE=1       - Enable fast mode
  RAY_SSH_MUX=0         - Disable SSH connection multiplexing
  RAY_PARALLEL_JOBS=N   - Set parallel job count
EOF
}

# 日志函数
log_message() {
    local level=$1
    local color=$2
    local msg=$3
    echo -e "${color}[$level]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $msg"
}

log_info()  { log_message "INFO" "$GREEN" "$1"; }
log_warn()  { log_message "WARN" "$YELLOW" "$1" >&2; }
log_error() { log_message "ERROR" "$RED" "$1" >&2; }
log_fatal() { log_message "FATAL" "$RED" "$1" >&2; exit 1; }
log_debug() { log_message "DEBUG" "$CYAN" "$1"; }

# --- 3. 参数解析 ---
AUTO_PORT=false
FORCE_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --node-list) NODE_LIST_FILE="$2"; shift 2 ;;
        --port) BASE_MASTER_PORT="$2"; shift 2 ;;
        --dashboard-port) BASE_DASHBOARD_PORT="$2"; shift 2 ;;
        --metrics-port) BASE_METRICS_PORT="$2"; shift 2 ;;
        --worker-port-start) MIN_WORKER_PORT="$2"; shift 2 ;;
        --worker-port-end) MAX_WORKER_PORT="$2"; shift 2 ;;
        --npus-per-node) NPUS_PER_NODE="$2"; shift 2 ;;
        --wait-time) WAIT_TIME="$2"; shift 2 ;;
        --auto-port) AUTO_PORT=true; shift ;;
        --force-cleanup) FORCE_CLEANUP=true; shift ;;
        --fast-mode) FAST_MODE=true; shift ;;
        --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        -*) log_fatal "Unknown option '$1'. Use --help for usage." ;;
        *) log_fatal "Unexpected argument '$1'. Use --help for usage." ;;
    esac
done

if [[ "${RAY_PORT_AUTO:-0}" == "1" ]]; then
    AUTO_PORT=true
fi

# --- 4. 快速端口检查（批量并行） ---

check_port_available() {
    local port=$1
    local node=$2
    
    if [[ "$node" == "localhost" ]] || [[ "$node" == "$(hostname)" ]]; then
        ! (ss -tuln 2>/dev/null || netstat -tuln 2>/dev/null) | grep -q ":${port} "
    else
        ssh $SSH_OPTS -q "$node" "! (ss -tuln 2>/dev/null | grep -q ':${port} ' || netstat -tuln 2>/dev/null | grep -q ':${port} ')" 2>/dev/null
    fi
}

find_available_port() {
    local base_port=$1
    local node=$2
    local max_offset=${3:-$PORT_SEARCH_RANGE}
    
    for offset in $(seq 0 $max_offset); do
        local port=$((base_port + offset))
        if check_port_available "$port" "$node"; then
            echo "$port"
            return 0
        fi
    done
    return 1
}

# 并行清理
cleanup_ray_on_node() {
    local node=$1
    ssh $SSH_OPTS -q "$node" '
        ray stop -f 2>/dev/null || true
        pkill -f "ray::" 2>/dev/null || true
        sleep 0.5
        pkill -9 -f "ray::" 2>/dev/null || true
    ' 2>/dev/null || true
}

# --- 5. 核心函数 ---

_remote_stop_ray() {
    set -euo pipefail
    ray stop -f >/dev/null 2>&1 || true
    pkill -f "ray::" 2>/dev/null || true
}

_remote_start_ray_head() {
    local port="$1"
    local dashboard_port="$2"
    local metrics_port="$3"
    local npus="$4"
    local min_worker_port="$5"
    local max_worker_port="$6"
    local node_ip="$7"

    set -euo pipefail
    local resources_json="{\"NPU\": ${npus}}"

    # 强制使用指定的节点 IP
    export RAY_NODE_IP_ADDRESS="${node_ip}"
    export VLLM_HOST_IP="${node_ip}"

    ray start --head \
        --port "${port}" \
        --node-ip-address "${node_ip}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${dashboard_port}" \
        --num-gpus="${npus}" \
        --resources="${resources_json}" \
        --metrics-export-port="${metrics_port}" \
        --min-worker-port="${min_worker_port}" \
        --max-worker-port="${max_worker_port}" \
        2>&1
}

_remote_start_ray_worker() {
    local master_addr="$1"
    local port="$2"
    local npus="$3"
    local min_worker_port="$4"
    local max_worker_port="$5"
    local node_ip="$6"

    set -euo pipefail
    local resources_json="{\"NPU\": ${npus}}"

    # 强制使用指定的节点 IP
    export RAY_NODE_IP_ADDRESS="${node_ip}"
    export VLLM_HOST_IP="${node_ip}"

    ray start --address "${master_addr}:${port}" \
        --node-ip-address "${node_ip}" \
        --num-gpus="${npus}" \
        --resources="${resources_json}" \
        --metrics-export-port=0 \
        --min-worker-port="${min_worker_port}" \
        --max-worker-port="${max_worker_port}" \
        2>&1
}

remote_exec_func() {
    local node="$1"
    local func_name="$2"
    shift 2
    local args=("$@")

    local func_code call_code
    func_code="$(declare -f "$func_name")"

    local args_str=""
    for arg in "${args[@]}"; do
        args_str+=" '${arg}'"
    done
    call_code="${func_name}${args_str}"

    local ssh_cmd="cd '${PROJECT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    echo "cd '${PROJECT_DIR}' && source set_env.sh; ${func_code}; ${call_code}" \
        | ssh $SSH_OPTS "$node" "$ssh_cmd"
}

stop_ray_parallel() {
    local nodes=("$@")
    local pids=()
    for node in "${nodes[@]}"; do
        (remote_exec_func "$node" _remote_stop_ray >/dev/null 2>&1 || true) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done
}

start_ray_node() {
    local node=$1
    local is_head=$2
    local node_ip=$3

    if $is_head; then
        log_info "[HEAD] Starting Ray head on $node (IP: $node_ip)..."
        remote_exec_func "$node" _remote_start_ray_head "$MASTER_PORT" "$DASHBOARD_PORT" "$METRICS_PORT" "$NPUS_PER_NODE" "$MIN_WORKER_PORT" "$MAX_WORKER_PORT" "$node_ip"
    else
        log_info "[WORKER] Starting Ray worker on $node (IP: $node_ip)..."
        remote_exec_func "$node" _remote_start_ray_worker "$MASTER_ADDR" "$MASTER_PORT" "$NPUS_PER_NODE" "$MIN_WORKER_PORT" "$MAX_WORKER_PORT" "$node_ip"
    fi
}

# 快速检测 head 就绪
check_head_ready() {
    local node=$1
    local max_wait=${2:-30}
    local elapsed=0
    
    while [[ $elapsed -lt $max_wait ]]; do
        if ssh $SSH_OPTS "$node" "docker exec \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -c 'ray status'" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
        elapsed=$(echo "$elapsed + 0.2" | bc 2>/dev/null || echo "$((elapsed + 1))")
    done
    return 1
}

# --- 6. 端口分配 ---
log_info "============================================="
log_info "Ray Cluster Fast Setup"
log_info "============================================="

if [[ -z "$NODE_LIST_FILE" ]] || [[ ! -f "$NODE_LIST_FILE" ]]; then
    log_fatal "Node list file is required and must exist."
fi

mapfile -t NODE_HOSTS < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")
if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    log_fatal "Node list is empty or contains no valid hosts."
fi

MASTER_ADDR="${MASTER_NODE:-${NODE_HOSTS[0]}}"

if [[ "$AUTO_PORT" == true ]]; then
    log_info "Auto-port mode enabled..."
    MASTER_PORT=$(find_available_port "$BASE_MASTER_PORT" "$MASTER_ADDR")
    [[ -z "$MASTER_PORT" ]] && log_fatal "No available master port"
    
    DASHBOARD_PORT=$(find_available_port "$BASE_DASHBOARD_PORT" "$MASTER_ADDR")
    [[ -z "$DASHBOARD_PORT" ]] && log_fatal "No available dashboard port"
    
    METRICS_PORT=$(find_available_port "$BASE_METRICS_PORT" "$MASTER_ADDR")
    [[ -z "$METRICS_PORT" ]] && log_fatal "No available metrics port"
    
    log_info "Ports: master=$MASTER_PORT dashboard=$DASHBOARD_PORT metrics=$METRICS_PORT"
else
    MASTER_PORT="$BASE_MASTER_PORT"
    DASHBOARD_PORT="$BASE_DASHBOARD_PORT"
    METRICS_PORT="$BASE_METRICS_PORT"
fi

# --- 7. 快速预检 ---
log_info "Running pre-checks..."
start_time=$(date +%s.%N 2>/dev/null || date +%s)

ssh_failed=()
for node in "${NODE_HOSTS[@]}"; do
    if ! ssh $SSH_OPTS -q "$node" "exit" 2>/dev/null; then
        ssh_failed+=("$node")
    fi
done &
ssh_check_pid=$!

# 端口冲突快速检查
if ! check_port_available "$MASTER_PORT" "$MASTER_ADDR"; then
    log_error "Port $MASTER_PORT is in use on $MASTER_ADDR"
    [[ "$AUTO_PORT" == false ]] && log_fatal "Use --auto-port to find available ports"
fi

wait $ssh_check_pid
[ ${#ssh_failed[@]} -gt 0 ] && log_fatal "SSH failed to: ${ssh_failed[*]}"

end_time=$(date +%s.%N 2>/dev/null || date +%s)
pre_check_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "1")
log_info "Pre-checks passed in ${pre_check_time}s"

# --- 8. 清理与启动 ---
NUM_NODES=${#NODE_HOSTS[@]}
WORKERS=()
for node in "${NODE_HOSTS[@]}"; do
    [[ "$node" != "$MASTER_ADDR" ]] && WORKERS+=("$node")
done

log_info "Configuration: $NUM_NODES nodes, ${#WORKERS[@]} workers"
log_info "Worker port range: ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"

# 并行清理
log_info "Cleaning up existing Ray processes..."
stop_ray_parallel "${NODE_HOSTS[@]}"

# 启动集群
log_info "Starting Ray cluster..."
cluster_start=$(date +%s.%N 2>/dev/null || date +%s)

# 获取每个节点的 IP 地址
log_info "Resolving node IP addresses..."
declare -A NODE_IPS
for node in "${NODE_HOSTS[@]}"; do
    node_ip=$(ssh $SSH_OPTS "$node" '
        source "'"$SCRIPT_DIR"'"/set_env.sh 2>/dev/null
        if [ -n "${RAY_NODE_IP_ADDRESS:-}" ]; then
            echo "$RAY_NODE_IP_ADDRESS"
        else
            ip -4 addr show ${GLOO_SOCKET_IFNAME:-eth0} 2>/dev/null | awk "/inet / {print \$2}" | cut -d/ -f1 | head -n1
        fi
    ' 2>/dev/null || echo "$node")
    NODE_IPS[$node]="$node_ip"
    log_info "  $node -> $node_ip"
done

HEAD_IP="${NODE_IPS[$MASTER_ADDR]}"

if ! start_ray_node "$MASTER_ADDR" true "$HEAD_IP"; then
    log_fatal "Failed to start Ray head on $MASTER_ADDR"
fi

# 快速轮询等待
if [[ "$WAIT_TIME" == "0" ]] || [[ "$WAIT_TIME" == "0.0" ]]; then
    log_info "Waiting for head node to be ready..."
    if ! check_head_ready "$MASTER_ADDR" 10; then
        log_warn "Head node may not be fully ready"
    fi
else
    log_info "Waiting ${WAIT_TIME}s..."
    sleep $WAIT_TIME
fi

# 并行启动 workers
success_count=1
failed_count=0
failed_nodes=()

if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Starting ${#WORKERS[@]} worker nodes in parallel..."
    
    # 使用 FIFO 控制并发
    exec 9>/tmp/ray_worker_flock
    
    # 存储子进程 PID
    declare -A worker_pids
    for worker in "${WORKERS[@]}"; do
        worker_ip="${NODE_IPS[$worker]}"
        (
            flock -n 9 || flock 9
            if start_ray_node "$worker" false "$worker_ip" >/dev/null 2>&1; then
                flock -u 9
                exit 0
            else
                flock -u 9
                exit 1
            fi
        ) &
        worker_pids[$worker]=$!
    done
    
    # 等待所有子进程完成并收集结果
    for worker in "${WORKERS[@]}"; do
        pid=${worker_pids[$worker]}
        if wait $pid; then
            ((success_count++))
            log_info "Worker $worker connected"
        else
            ((failed_count++))
            failed_nodes+=("$worker")
            log_error "Worker $worker failed"
        fi
    done
fi

cluster_end=$(date +%s.%N 2>/dev/null || date +%s)
cluster_time=$(echo "$cluster_end - $cluster_start" | bc 2>/dev/null || echo "5")

# --- 9. 报告 ---
echo ""
echo -e "${BLUE}=============================================${NC}"
if [ $failed_count -eq 0 ]; then
    log_message "SUCCESS" "$GREEN" "Cluster started in ${cluster_time}s!"
else
    log_warn "Started with $failed_count failed: ${failed_nodes[*]}"
fi

log_info "Dashboard: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "Nodes: $success_count/$NUM_NODES"
echo -e "${BLUE}=============================================${NC}"

# 后台状态显示
(
    sleep 1
    ssh $SSH_OPTS "$MASTER_ADDR" "cd '${PROJECT_DIR}' && source set_env.sh && docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -c 'ray status'" 2>/dev/null || true
) &

# 保存端口配置
PORT_CONFIG_FILE="/tmp/ray_cluster_ports_$(echo $MASTER_ADDR | tr '.' '_').sh"
cat > "$PORT_CONFIG_FILE" << EOF
export RAY_PORT=${MASTER_PORT}
export RAY_DASHBOARD_PORT=${DASHBOARD_PORT}
export RAY_METRICS_PORT=${METRICS_PORT}
EOF
log_info "Port config saved to: $PORT_CONFIG_FILE"
