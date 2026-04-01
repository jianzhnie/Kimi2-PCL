#!/usr/bin/env bash

# =================================================================
# Ray Cluster Launcher (Optimized)
# -----------------------------------------------------------------
# 目的: 启动多节点 Ray 集群，支持配置 NPU 资源。
# 依赖: 所有节点上都安装了 Ray 并配置了无密码 SSH 登录。
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 0. 加载环境变量 ---
ENV_FILE="${SCRIPT_DIR}/set_env.sh"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
else
    echo "ERROR: Environment config file not found: ${ENV_FILE}" >&2
    exit 1
fi

# --- 1. 默认配置与常量 ---
PROJECT_DIR="${SCRIPT_DIR}"
MASTER_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
WAIT_TIME=1
NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
CONTAINER="${CONTAINER_NAME:-vllm-ascend-env-a3}"
RETRY_COUNT=2
SSH_TIMEOUT=10
CLEAN_ONLY=false
CLEAN_START=false

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

Starts a multi-node Ray cluster, assigning NPU resources.

Options:
  --project-dir DIR     Project directory containing 'set_env.sh' (default: $PROJECT_DIR)
  --node-list FILE      File containing list of nodes (default: $NODE_LIST_FILE)
  --port PORT           Ray master port (default: $MASTER_PORT)
  --dashboard-port PORT Dashboard port (default: $DASHBOARD_PORT)
  --npus-per-node NUM   Number of NPUs per node (default: $NPUS_PER_NODE)
  --wait-time SEC       Wait time for head node initialization (default: $WAIT_TIME)
  --retry NUM           Number of retries for worker nodes (default: 2)
  --timeout SEC         SSH connection timeout in seconds (default: 10)
  --clean               Stop Ray on all nodes and exit (cleanup only)
  --clean-start         Stop Ray on all nodes before starting cluster
  --help                Show this help message

Environment variables (from set_env.sh or override):
  RAY_PORT              Ray master port
  RAY_DASHBOARD_PORT    Ray dashboard port
  MASTER_NODE           Explicitly set master node (overrides node-list first entry)
  NODES_FILE            Path to node list file
  CONTAINER_NAME        Docker container name (default: vllm-ascend-env-a3)
  SSH_OPTS              SSH options for connections
EOF
}

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

# --- 3. 工具函数 ---
validate_port() {
    local port="$1"
    local name="$2"
    if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        log_fatal "Invalid $name: '$port'. Must be an integer between 1 and 65535."
    fi
}

validate_positive_int() {
    local value="$1"
    local name="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        log_fatal "Invalid $name: '$value'. Must be a positive integer."
    fi
}

# 构建 SSH 选项
get_ssh_opts() {
    local timeout="${1:-$SSH_TIMEOUT}"
    echo "${SSH_OPTS} -o ConnectTimeout=${timeout}"
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-30}"
    local interval=1
    local elapsed=0
    local ssh_opts
    ssh_opts=$(get_ssh_opts 5)

    while [ $elapsed -lt "$timeout" ]; do
        # shellcheck disable=SC2086
        if ssh $ssh_opts -q \
            "$host" "bash -c 'exec 3<>/dev/tcp/localhost/$port' 2>/dev/null"; then
            return 0
        fi
        sleep $interval
        ((elapsed += interval))
    done
    return 1
}

# --- 4. 参数解析 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --node-list) NODE_LIST_FILE="$2"; shift 2 ;;
        --port) MASTER_PORT="$2"; shift 2 ;;
        --dashboard-port) DASHBOARD_PORT="$2"; shift 2 ;;
        --npus-per-node) NPUS_PER_NODE="$2"; shift 2 ;;
        --wait-time) WAIT_TIME="$2"; shift 2 ;;
        --retry) RETRY_COUNT="$2"; shift 2 ;;
        --timeout) SSH_TIMEOUT="$2"; shift 2 ;;
        --clean) CLEAN_ONLY=true; shift ;;
        --clean-start) CLEAN_START=true; shift ;;
        --help|-h) usage; exit 0 ;;
        -*) log_fatal "Unknown option '$1'. Use --help for usage." ;;
        *) log_fatal "Unexpected argument '$1'. Use --help for usage." ;;
    esac
done

# 验证参数
validate_port "$MASTER_PORT" "master port"
validate_port "$DASHBOARD_PORT" "dashboard port"
validate_positive_int "$NPUS_PER_NODE" "NPUs per node"
validate_positive_int "$WAIT_TIME" "wait time"
validate_positive_int "$RETRY_COUNT" "retry count"
validate_positive_int "$SSH_TIMEOUT" "SSH timeout"

# 更新容器名称（优先使用环境变量，然后是指定值）
CONTAINER="${CONTAINER_NAME:-$CONTAINER}"

# --- 5. 核心函数：远程执行命令 ---

_remote_stop_ray() {
    set -euo pipefail
    ray stop -f >/dev/null 2>&1 || true
}

_remote_start_ray_head() {
    local port="$1"
    local dashboard_port="$2"
    local npus="$3"
    local node_ip="${4:-}"

    set -euo pipefail
    local resources_json="{\"NPU\": ${npus}}"
    local node_ip_flag=""
    [[ -n "$node_ip" ]] && node_ip_flag="--node-ip-address ${node_ip}"

    # 使用明确的端口范围避免冲突
    # metrics-export-port: 使用高位端口 50000，避免与其他组件冲突
    # worker 端口范围: 40000-49999，与 Ray 内部随机端口范围错开
    # Ray 内部组件端口通常在 10000-30000 范围随机分配
    # shellcheck disable=SC2086
    ray start --head \
        --port "${port}" \
        ${node_ip_flag} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${dashboard_port}" \
        --num-gpus="${npus}" \
        --resources="${resources_json}" \
        --metrics-export-port=50000 \
        --min-worker-port=40000 \
        --max-worker-port=49999
}

_remote_start_ray_worker() {
    local master_addr="$1"
    local port="$2"
    local npus="$3"
    local node_ip="${4:-}"

    set -euo pipefail
    local resources_json="{\"NPU\": ${npus}}"
    local node_ip_flag=""
    [[ -n "$node_ip" ]] && node_ip_flag="--node-ip-address ${node_ip}"

    # 使用与 head 节点一致的端口配置
    # shellcheck disable=SC2086
    ray start --address "${master_addr}:${port}" \
        ${node_ip_flag} \
        --num-gpus="${npus}" \
        --resources="${resources_json}" \
        --metrics-export-port=50000 \
        --min-worker-port=40000 \
        --max-worker-port=49999
}

# 包装器：将本地函数发送到远端执行
remote_exec_func() {
    local node="$1"
    local func_name="$2"
    shift 2
    local args=("$@")

    local func_code call_code args_str
    func_code="$(declare -f "$func_name")"

    # 转义参数
    args_str=""
    for arg in "${args[@]}"; do
        args_str+=" $(printf '%q' "$arg")"
    done
    call_code="${func_name}${args_str}"

    local ssh_opts proj_dir_escaped
    ssh_opts=$(get_ssh_opts)
    proj_dir_escaped=$(printf '%q' "$PROJECT_DIR")

    # 在远端执行：source 环境变量，然后通过 docker exec 执行函数
    local remote_setup="cd ${proj_dir_escaped} && source set_env.sh"
    local docker_exec="docker exec -i \"${CONTAINER}\" bash -s"
    local container_cmds="cd ${proj_dir_escaped} && source set_env.sh; ${func_code}; ${call_code}"

    # shellcheck disable=SC2086
    echo "$container_cmds" | ssh $ssh_opts "$node" "$remote_setup && $docker_exec"
}

stop_ray_node() {
    local node=$1
    log_info "Stopping existing Ray on $node..."
    remote_exec_func "$node" _remote_stop_ray || true
}

# 清理 Ray 进程的远程函数
_remote_kill_ray_processes() {
    set -euo pipefail
    # 强制终止所有 Ray 相关进程
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "ray start" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    pkill -9 -f "redis-server" 2>/dev/null || true
    pkill -9 -f "dashboard" 2>/dev/null || true
    sleep 1
}

# 在所有节点上清理 Ray
cleanup_all_nodes() {
    local nodes=("$@")
    log_info "Cleaning up Ray processes on ${#nodes[@]} nodes..."
    
    local pids=()
    local failed_nodes=()
    
    for node in "${nodes[@]}"; do
        (
            log_info "Cleaning up $node..."
            # 先尝试优雅停止
            remote_exec_func "$node" _remote_stop_ray 2>/dev/null || true
            # 再强制清理残留进程
            remote_exec_func "$node" _remote_kill_ray_processes 2>/dev/null || true
        ) &
        pids+=($!)
    done
    
    # 等待所有清理完成
    local failed=0
    for i in "${!pids[@]}"; do
        if ! wait ${pids[$i]}; then
            ((failed++))
        fi
    done
    
    if [ $failed -gt 0 ]; then
        log_warn "Cleanup completed with $failed node(s) having issues."
    else
        log_info "Cleanup completed successfully on all nodes."
    fi
    
    # 等待一下确保端口释放
    log_info "Waiting 3 seconds for port release..."
    sleep 3
}

start_ray_node() {
    local node=$1
    local is_head=$2
    local node_ip="${RAY_NODE_IP_ADDRESS:-}"

    stop_ray_node "$node"

    if [ "$is_head" -eq 1 ]; then
        log_info "[HEAD] Starting Ray head on $node (Master: $MASTER_ADDR:$MASTER_PORT)"
        remote_exec_func "$node" _remote_start_ray_head \
            "$MASTER_PORT" "$DASHBOARD_PORT" "$NPUS_PER_NODE" "$node_ip"
    else
        log_info "[WORKER] Starting Ray worker on $node (Connecting to: $MASTER_ADDR:$MASTER_PORT)"
        remote_exec_func "$node" _remote_start_ray_worker \
            "$MASTER_ADDR" "$MASTER_PORT" "$NPUS_PER_NODE" "$node_ip"
    fi
}

start_ray_node_with_retry() {
    local node=$1
    local is_head=$2
    local max_retries=$3
    local attempt=1

    while [ $attempt -le "$max_retries" ]; do
        if start_ray_node "$node" "$is_head"; then
            return 0
        fi

        if [ $attempt -lt "$max_retries" ]; then
            log_warn "Retrying $node in 3 seconds... (attempt $attempt/$max_retries)"
            sleep 3
        fi
        ((attempt++))
    done
    return 1
}

# --- 6. 预检与节点列表处理 ---
[[ -f "$NODE_LIST_FILE" ]] || log_fatal "Node list file not found: $NODE_LIST_FILE"

mapfile -t NODE_HOSTS < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")
[ ${#NODE_HOSTS[@]} -eq 0 ] && log_fatal "Node list '$NODE_LIST_FILE' is empty or contains no valid hosts."

# 确定 master 节点
MASTER_ADDR="${MASTER_NODE:-${NODE_HOSTS[0]}}"
NUM_NODES=${#NODE_HOSTS[@]}

# 收集 Worker 节点
WORKERS=()
for node in "${NODE_HOSTS[@]}"; do
    [[ "$node" != "$MASTER_ADDR" ]] && WORKERS+=("$node")
done

# --- 处理清理选项 ---
if [ "$CLEAN_ONLY" = true ]; then
    log_info "Mode: CLEAN ONLY - Stopping Ray on all nodes and exiting..."
    cleanup_all_nodes "${NODE_HOSTS[@]}"
    log_info "Cleanup completed. Exiting."
    exit 0
fi

if [ "$CLEAN_START" = true ]; then
    log_info "Mode: CLEAN START - Cleaning up before starting cluster..."
    cleanup_all_nodes "${NODE_HOSTS[@]}"
fi

log_info "============================================="
log_info "Ray Cluster Setup Configuration"
log_info "============================================="
log_info "Total nodes: $NUM_NODES"
log_info "NPUs per node: $NPUS_PER_NODE"
log_info "Master IP: ${BLUE}$MASTER_ADDR${NC}"
log_info "Master port: $MASTER_PORT"
log_info "Dashboard port: $DASHBOARD_PORT"
log_info "Project directory: $PROJECT_DIR"
log_info "Container name: $CONTAINER"
log_info "Worker retry count: $RETRY_COUNT"
log_info "SSH timeout: ${SSH_TIMEOUT}s"
log_info "Worker nodes (${#WORKERS[@]}): ${WORKERS[*]:-None}"
log_info "============================================="

# --- 7. 并行预检 ---
log_info "Verifying SSH connections and environment on all nodes..."

errors=0
declare -a check_pids check_nodes

for node in "${NODE_HOSTS[@]}"; do
    (
        ssh_opts=$(get_ssh_opts)
        proj_dir_escaped=$(printf '%q' "$PROJECT_DIR")

        # 检查 SSH 连接
        # shellcheck disable=SC2086
        if ! ssh $ssh_opts -q "$node" "exit" 2>/dev/null; then
            log_error "SSH connection failed to $node"
            exit 1
        fi

        # 检查项目目录
        # shellcheck disable=SC2086
        if ! ssh $ssh_opts "$node" "[ -d ${proj_dir_escaped} ]" 2>/dev/null; then
            log_error "Project directory $PROJECT_DIR not found on $node"
            exit 1
        fi

        # 检查 set_env.sh
        # shellcheck disable=SC2086
        if ! ssh $ssh_opts "$node" "[ -f ${proj_dir_escaped}/set_env.sh ]" 2>/dev/null; then
            log_error "set_env.sh not found in $PROJECT_DIR on $node"
            exit 1
        fi

        # 检查容器是否运行
        # shellcheck disable=SC2086
        if ! ssh $ssh_opts "$node" "docker ps -q --filter name=^/${CONTAINER}\$ | grep -q ." 2>/dev/null; then
            log_error "Container '$CONTAINER' not running on $node"
            exit 1
        fi

        exit 0
    ) &
    check_pids+=($!)
    check_nodes+=("$node")
done

for i in "${!check_pids[@]}"; do
    # shellcheck disable=SC2086
    wait ${check_pids[$i]} || ((errors++))
done

[ $errors -gt 0 ] && log_fatal "Pre-checks failed with $errors errors."
log_info "All pre-checks passed."

# --- 8. 启动流程 ---
if ! start_ray_node "$MASTER_ADDR" 1; then
    log_fatal "Failed to start Ray head node on $MASTER_ADDR"
fi

log_info "Waiting for head node GCS port $MASTER_PORT to be ready..."
if wait_for_port "$MASTER_ADDR" "$MASTER_PORT" 30; then
    log_info "Head node is ready."
else
    log_warn "GCS port check timed out, proceeding anyway..."
fi

if [ "$WAIT_TIME" -gt 0 ]; then
    log_info "Waiting additional ${WAIT_TIME}s..."
    sleep "$WAIT_TIME"
fi

success_count=1
failed_count=0
declare -a failed_nodes

if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Starting ${#WORKERS[@]} worker nodes in parallel (with $RETRY_COUNT retries)..."
    declare -a pids node_names

    for worker in "${WORKERS[@]}"; do
        start_ray_node_with_retry "$worker" 0 "$RETRY_COUNT" &
        pids+=($!)
        node_names+=("$worker")
    done

    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        node=${node_names[$i]}
        # shellcheck disable=SC2086
        if wait $pid; then
            ((success_count++))
            log_info "Worker node $node connected successfully."
        else
            log_error "Worker node $node failed to connect."
            failed_nodes+=("$node")
            ((failed_count++))
        fi
    done
else
    log_info "No worker nodes defined. Starting single-node cluster."
fi

# --- 9. 最终报告 ---
echo ""
echo -e "${BLUE}=============================================${NC}"
if [ $failed_count -eq 0 ]; then
    log_message "SUCCESS" "$GREEN" "Ray cluster setup completed!"
else
    log_warn "Ray cluster setup completed with $failed_count worker node(s) failed!"
    log_info "Failed nodes: ${failed_nodes[*]}"
fi

log_info "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "Total nodes: $NUM_NODES ($success_count nodes running Ray)"
echo -e "${BLUE}=============================================${NC}"

# 显示 Ray 状态
echo ""
echo -e "${BLUE}=============================================${NC}"
log_info "Displaying Ray status..."

ssh_opts=$(get_ssh_opts)
proj_dir_escaped=$(printf '%q' "$PROJECT_DIR")

remote_setup="cd ${proj_dir_escaped} && source set_env.sh"
docker_exec="docker exec -i ${CONTAINER} bash -s"
status_cmd="cd ${proj_dir_escaped} && source set_env.sh && ray status"

# shellcheck disable=SC2086
timeout 30 ssh $ssh_opts "$MASTER_ADDR" \
    "$remote_setup && $docker_exec" <<< "$status_cmd" 2>/dev/null || {
    log_warn "Failed to retrieve Ray status. Cluster may still be initializing."
}
echo -e "${BLUE}=============================================${NC}"

# 根据失败情况返回退出码
[ $failed_count -gt 0 ] && exit 1
exit 0
