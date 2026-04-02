#!/usr/bin/env bash

# =================================================================
# Ray Cluster Launcher (Fast Mode)
# -----------------------------------------------------------------
# 目的: 快速启动多节点 Ray 集群，支持配置 NPU 资源。
# 优化: SSH 连接复用、并行预检、快速轮询检测
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
MASTER_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
NPUS_PER_NODE=8
WAIT_TIME="${RAY_WAIT_TIME:-0.5}"  # 减少默认等待时间
NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
FAST_MODE="${RAY_FAST_MODE:-true}"  # 快速模式，跳过非关键检查
PARALLEL_JOBS="${RAY_PARALLEL_JOBS:-8}"  # 并行任务数

# SSH 连接复用配置
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if [[ "${RAY_SSH_MUX:-true}" == "true" ]]; then
    SSH_OPTS="${SSH_OPTS} -o ControlMaster=auto -o ControlPersist=60s -o ControlPath=/tmp/ssh_mux_%h_%p_%r"
fi

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- 2. 帮助信息与日志函数 ---
usage() {
    cat <<EOF
Usage: $0 [options]

Starts a multi-node Ray cluster with optimized speed.

Options:
  --project-dir DIR     Project directory containing 'set_env.sh' (default: $PROJECT_DIR)
  --node-list FILE      File containing list of nodes (default: $NODE_LIST_FILE)
  --port PORT           Ray master port (default: $MASTER_PORT)
  --dashboard-port PORT Dashboard port (default: $DASHBOARD_PORT)
  --npus-per-node NUM   Number of NPUs per node (default: $NPUS_PER_NODE)
  --wait-time SEC       Wait time for head node initialization (default: $WAIT_TIME)
  --fast-mode           Skip non-critical checks for faster startup
  --no-fast-mode        Run all checks (slower but more thorough)
  --parallel-jobs NUM   Number of parallel jobs (default: $PARALLEL_JOBS)
  --help                Show this help message

Environment Variables:
  RAY_FAST_MODE=1       - Enable fast mode (skip non-critical checks)
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

# --- 3. 参数解析 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --node-list) NODE_LIST_FILE="$2"; shift 2 ;;
        --port) MASTER_PORT="$2"; shift 2 ;;
        --dashboard-port) DASHBOARD_PORT="$2"; shift 2 ;;
        --npus-per-node) NPUS_PER_NODE="$2"; shift 2 ;;
        --wait-time) WAIT_TIME="$2"; shift 2 ;;
        --fast-mode) FAST_MODE=true; shift ;;
        --no-fast-mode) FAST_MODE=false; shift ;;
        --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        -*) log_fatal "Unknown option '$1'. Use --help for usage." ;;
        *) log_fatal "Unexpected argument '$1'. Use --help for usage." ;;
    esac
done

# --- 4. 核心函数：远程执行命令 ---

# 批量检查节点（并行）
check_nodes_parallel() {
    local check_type=$1
    shift
    local nodes=("$@")
    local failed=0
    local pids=()
    local results=()
    
    for node in "${nodes[@]}"; do
        (
            case $check_type in
                ssh)
                    ssh $SSH_OPTS -q "$node" "exit" 2>/dev/null || echo "SSH_FAILED:$node"
                    ;;
                dir)
                    ssh $SSH_OPTS "$node" "[ -d \"$PROJECT_DIR\" ]" 2>/dev/null || echo "DIR_FAILED:$node"
                    ;;
                env)
                    ssh $SSH_OPTS "$node" "[ -f \"$PROJECT_DIR/set_env.sh\" ]" 2>/dev/null || echo "ENV_FAILED:$node"
                    ;;
            esac
        ) &
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        wait $pid || true
    done
}

_remote_stop_ray() {
    set -euo pipefail
    ray stop -f >/dev/null 2>&1 || true
    pkill -f "ray::" 2>/dev/null || true
}

_remote_start_ray_head() {
    local port="$1"
    local dashboard_port="$2"
    local npus="$3"
    local master_addr="$4"
    local node_ip="$5"

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
        --metrics-export-port=0 \
        --min-worker-port=30000 \
        --max-worker-port=35000 \
        --include-dashboard=true \
        2>&1
}

_remote_start_ray_worker() {
    local master_addr="$1"
    local port="$2"
    local npus="$3"
    local node_ip="$4"

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
        --min-worker-port=30000 \
        --max-worker-port=35000 \
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

# 并行停止 Ray
stop_ray_parallel() {
    local nodes=("$@")
    local pids=()
    for node in "${nodes[@]}"; do
        (
            remote_exec_func "$node" _remote_stop_ray >/dev/null 2>&1 || true
        ) &
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
        if ! remote_exec_func "$node" _remote_start_ray_head "$MASTER_PORT" "$DASHBOARD_PORT" "$NPUS_PER_NODE" "$MASTER_ADDR" "$node_ip"; then
            log_error "Failed to start Ray head on node $node"
            return 1
        fi
    else
        log_info "[WORKER] Starting Ray worker on $node (IP: $node_ip)..."
        if ! remote_exec_func "$node" _remote_start_ray_worker "$MASTER_ADDR" "$MASTER_PORT" "$NPUS_PER_NODE" "$node_ip"; then
            log_error "Failed to start Ray worker on node $node"
            return 1
        fi
    fi
    return 0
}

# 快速检测 head 节点是否就绪
check_head_ready() {
    local node=$1
    local max_wait=${2:-30}
    local interval=${3:-0.2}
    local elapsed=0
    
    while [[ $elapsed -lt $max_wait ]]; do
        if ssh $SSH_OPTS "$node" "docker exec \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -c 'ray status'" >/dev/null 2>&1; then
            return 0
        fi
        sleep $interval
        elapsed=$(echo "$elapsed + $interval" | bc 2>/dev/null || echo "$((elapsed + 1))")
    done
    return 1
}

# --- 5. 节点列表处理 ---

if [[ -z "$NODE_LIST_FILE" ]]; then
    log_fatal "Node list file is required."
fi

if [ ! -f "$NODE_LIST_FILE" ]; then
    log_fatal "Node list file '$NODE_LIST_FILE' does not exist!"
fi

mapfile -t NODE_HOSTS < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")

if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    log_fatal "Node list '$NODE_LIST_FILE' is empty or contains no valid hosts."
fi

MASTER_ADDR="${MASTER_NODE:-${NODE_HOSTS[0]}}"
NUM_NODES=${#NODE_HOSTS[@]}

WORKERS=()
for node in "${NODE_HOSTS[@]}"; do
    if [[ "$node" != "$MASTER_ADDR" ]]; then
        WORKERS+=("$node")
    fi
done

log_info "============================================="
log_info "Ray Cluster Fast Setup"
log_info "============================================="
log_info "Total nodes: $NUM_NODES"
log_info "NPUs per node: $NPUS_PER_NODE"
log_info "Master IP: ${BLUE}$MASTER_ADDR${NC}"
log_info "Master port: $MASTER_PORT"
log_info "Dashboard port: $DASHBOARD_PORT"
log_info "Fast mode: $FAST_MODE"
log_info "============================================="

# --- 6. 预检（并行化） ---
log_info "Running pre-checks..."
start_time=$(date +%s.%N 2>/dev/null || date +%s)

# SSH 连接检查（并行）
ssh_failed=()
for node in "${NODE_HOSTS[@]}"; do
    if ! ssh $SSH_OPTS -q "$node" "exit" 2>/dev/null; then
        ssh_failed+=("$node")
    fi
done &
ssh_check_pid=$!

# 目录检查（可选，fast mode 跳过）
if [[ "$FAST_MODE" != "true" ]]; then
    dir_failed=()
    env_failed=()
    for node in "${NODE_HOSTS[@]}"; do
        if ! ssh $SSH_OPTS "$node" "[ -d \"$PROJECT_DIR\" ]" 2>/dev/null; then
            dir_failed+=("$node")
        elif ! ssh $SSH_OPTS "$node" "[ -f \"$PROJECT_DIR/set_env.sh\" ]" 2>/dev/null; then
            env_failed+=("$node")
        fi
    done &
    dir_check_pid=$!
fi

wait $ssh_check_pid
if [ ${#ssh_failed[@]} -gt 0 ]; then
    log_fatal "SSH connection failed to: ${ssh_failed[*]}"
fi

if [[ "$FAST_MODE" != "true" ]]; then
    wait $dir_check_pid 2>/dev/null || true
    if [ ${#dir_failed[@]} -gt 0 ]; then
        log_fatal "Project directory not found on: ${dir_failed[*]}"
    fi
    if [ ${#env_failed[@]} -gt 0 ]; then
        log_fatal "set_env.sh not found on: ${env_failed[*]}"
    fi
fi

end_time=$(date +%s.%N 2>/dev/null || date +%s)
pre_check_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "1")
log_info "Pre-checks passed in ${pre_check_time}s"

# --- 7. 快速清理现有 Ray（并行） ---
log_info "Stopping existing Ray processes..."
stop_ray_parallel "${NODE_HOSTS[@]}"

# --- 8. 启动流程 ---
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

# 启动 Head 节点
if ! start_ray_node "$MASTER_ADDR" true "$HEAD_IP"; then
    log_fatal "Failed to start Ray head node on $MASTER_ADDR"
fi

# 快速轮询检测 head 就绪（替代固定等待）
if [[ "$WAIT_TIME" == "0" ]] || [[ "$WAIT_TIME" == "0.0" ]]; then
    log_info "Waiting for head node to be ready..."
    if ! check_head_ready "$MASTER_ADDR" 10 0.2; then
        log_warn "Head node may not be fully ready, continuing anyway..."
    fi
else
    log_info "Waiting ${WAIT_TIME}s for head node..."
    sleep $WAIT_TIME
fi

# 并行启动 Worker 节点
success_count=1
failed_count=0
failed_nodes=()

if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Starting ${#WORKERS[@]} worker nodes in parallel (jobs: $PARALLEL_JOBS)..."
    
    # 使用文件描述符控制并发
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
            log_info "Worker $worker connected."
        else
            ((failed_count++))
            failed_nodes+=("$worker")
            log_error "Worker $worker failed."
        fi
    done
else
    log_info "No worker nodes defined. Starting single-node cluster."
fi

cluster_end=$(date +%s.%N 2>/dev/null || date +%s)
cluster_time=$(echo "$cluster_end - $cluster_start" | bc 2>/dev/null || echo "5")

# --- 9. 最终报告 ---
echo ""
echo -e "${BLUE}=============================================${NC}"
if [ $failed_count -eq 0 ]; then
    log_message "SUCCESS" "$GREEN" "Ray cluster started in ${cluster_time}s!"
else
    log_warn "Ray cluster started with $failed_count worker node(s) failed!"
    log_info "Failed nodes: ${failed_nodes[*]}"
fi

log_info "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "Total nodes: $NUM_NODES ($success_count nodes running Ray)"
echo -e "${BLUE}=============================================${NC}"

# 后台显示状态（不阻塞）
(
    sleep 1
    ssh $SSH_OPTS "$MASTER_ADDR" \
        "cd '${PROJECT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" \
        bash -c \"ray status 2>/dev/null || echo 'Cluster still initializing...'\"" 2>/dev/null || true
) &
