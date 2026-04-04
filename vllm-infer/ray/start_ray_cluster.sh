#!/usr/bin/env bash

# =================================================================
# Ray Cluster Launcher (Optimized)
# -----------------------------------------------------------------
# 目的: 启动多节点 Ray 集群，支持配置 NPU 资源。
# 依赖: 所有节点上都安装了 Ray 并配置了无密码 SSH 登录。
# =================================================================

set -euo pipefail  # 严格模式：遇到错误退出，未定义变量报错，管道错误退出

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/set_env.sh"
NODE_LIST_FILE="${NODES_FILE:-${PROJECT_DIR}/node_list.txt}"

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color


# --- 2. 帮助信息与日志函数 ---
usage() {
    cat <<EOF
Usage: $0 [options]

Starts a multi-node Ray cluster, assigning NPU resources.

Options:
  --node-list FILE      File containing list of nodes (default: $NODE_LIST_FILE)
  --help                Show this help message
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


# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
else
  echo "[ERROR] 环境配置文件未找到: ${ENV_FILE}" >&2
  exit 1
fi

# -------------------------------------------
# --- 2. 预检与节点列表处理 ---

if [[ -z "$NODE_LIST_FILE" ]]; then
    log_fatal "Node list file is required."
fi

if [ ! -f "$NODE_LIST_FILE" ]; then
    log_fatal "Node list file '$NODE_LIST_FILE' does not exist!"
fi

# 读取节点列表
mapfile -t NODE_HOSTS < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")

if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    log_fatal "Node list '$NODE_LIST_FILE' is empty or contains no valid hosts."
fi

# -----------------------------------------------

# --- 1. 默认配置与常量 ---
# 覆盖或使用环境变量，如果未定义则使用默认值
PROJECT_DIR="${PROJECT_DIR}"
NUM_NODES=${#NODE_HOSTS[@]}
MASTER_ADDR="${MASTER_ADDR:-${NODE_HOSTS[0]}}"     # Ray Head node IP
MASTER_PORT="${MASTER_PORT:-29500}"                 # Ray head node 端口
DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"       # Ray 仪表盘端口
NPUS_PER_NODE=8                                 # 每个节点的 NPU 数量
WAIT_TIME=1                                     # 等待头节点初始化的时间 (秒)

# 收集 Worker 节点
WORKERS=()
for node in "${NODE_HOSTS[@]}"; do
    if [[ "$node" != "$MASTER_ADDR" ]]; then
        WORKERS+=("$node")
    fi
done

# --- 4. 核心函数：远程执行命令 ---
# 使用 declare -f 序列化执行，避免引号转义问题，并复用 set_env.sh

_remote_stop_ray() {
    set -euo pipefail
    ray stop -f >/dev/null 2>&1 || true
}

_remote_start_ray_head() {
    local master_address="$1"
    local master_port="$2"
    local dashboard_port="$3"
    local num_gpus="$4"

    set -euo pipefail
    local resources_json="{\"NPU\": ${num_gpus}}"

    ray start --head \
        --node-ip-address=${master_address} \
        --port "${master_port}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${dashboard_port}" \
        --num-gpus="${num_gpus}" \
        --resources="${resources_json}" 
}

_remote_start_ray_worker() {
    local master_address="$1"
    local master_port="$2"
    local node_ip_adress="$3"
    local num_gpus="$4"

    set -euo pipefail
    local resources_json="{\"NPU\": ${num_gpus}}"

    ray start \
        --address "${master_address}:${master_port}" \
        --node-ip-address="${node_ip_adress}" \
        --num-gpus="${num_gpus}" \
        --resources="${resources_json}" 
}

# 包装器：将本地函数发送到远端执行，并预先 source 环境配置
# 并且通过 docker exec 进入容器执行
remote_exec_func() {
    local node="$1"
    local func_name="$2"
    shift 2
    local args=("$@")

    local func_code call_code
    func_code="$(declare -f "$func_name")"

    # 构造带参数的调用字符串
    local args_str=""
    for arg in "${args[@]}"; do
        args_str+=" '${arg}'"
    done
    call_code="${func_name}${args_str}"

    # 在远端宿主机：进入目录，source 环境变量（获取 CONTAINER_NAME 等）
    # 然后将代码通过管道喂给 docker exec
    local ssh_cmd="cd '${PROJECT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    # 容器内执行逻辑：
    # 1. source 项目目录下的 set_env.sh，确保容器内加载了 Ascend 环境变量及 Ray 所需变量
    # 2. 执行传入的函数
    echo "cd '${PROJECT_DIR}' && source set_env.sh; ${func_code}; ${call_code}" \
        | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$node" "$ssh_cmd"
}

stop_ray_node() {
    local node=$1
    log_info "Stopping existing Ray on $node..."
    remote_exec_func "$node" _remote_stop_ray || true
}

start_ray_node() {
    local node=$1
    local is_head=$2

    stop_ray_node "$node"

    if $is_head; then
        log_info "[HEAD] Starting Ray head on $node (Master: $MASTER_ADDR$:$MASTER_PORT)..."
        if ! remote_exec_func "$node" _remote_start_ray_head "$MASTER_ADDR" "$MASTER_PORT" "$DASHBOARD_PORT" "$NPUS_PER_NODE"; then
            log_error "Failed to start Ray head on node $node"
            return 1
        fi
    else
        log_info "[WORKER] Starting Ray worker on $node (Connecting to: $MASTER_ADDR:$MASTER_PORT)..."
        if ! remote_exec_func "$node" _remote_start_ray_worker "$MASTER_ADDR" "$MASTER_PORT" "$node" "$NPUS_PER_NODE"; then
            log_error "Failed to start Ray worker on node $node"
            return 1
        fi
    fi
    return 0
}


log_info "============================================="
log_info "Ray Cluster Setup Configuration"
log_info "============================================="
log_info "Total nodes: $NUM_NODES"
log_info "NPUs per node: $NPUS_PER_NODE"
log_info "Master IP: ${BLUE}$MASTER_ADDR${NC}"
log_info "Master port: $MASTER_PORT"
log_info "Dashboard port: $DASHBOARD_PORT"
log_info "Project directory: $PROJECT_DIR"
log_info "Worker nodes (${#WORKERS[@]}): ${WORKERS[*]:-None}"
log_info "============================================="

log_info "Verifying SSH connections and project directories on all nodes..."
errors=0
for node in "${NODE_HOSTS[@]}"; do
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -q "$node" "exit" >/dev/null 2>&1; then
        log_error "SSH connection failed to $node. Ensure SSH keys are set up correctly."
        ((errors++))
        continue
    fi
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$node" \
        "[ -d \"$PROJECT_DIR\" ]" >/dev/null 2>&1; then
        log_error "Project directory $PROJECT_DIR not found on node $node."
        ((errors++))
        continue
    fi
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$node" \
        "[ -f \"$PROJECT_DIR/set_env.sh\" ]" >/dev/null 2>&1; then
        log_error "set_env.sh not found in $PROJECT_DIR on node $node."
        ((errors++))
        continue
    fi
done

if [ $errors -gt 0 ]; then
    log_fatal "Pre-checks failed with $errors errors. Please fix them before continuing."
fi
log_info "All pre-checks passed."

# --- 6. 启动流程 ---
if ! start_ray_node "$MASTER_ADDR" true; then
    log_fatal "Failed to start Ray head node on $MASTER_ADDR"
fi

log_info "Waiting ${WAIT_TIME}s for head node to initialize..."
sleep $WAIT_TIME

success_count=1
failed_count=0
failed_nodes=()

if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Starting ${#WORKERS[@]} worker nodes in parallel..."
    pids=()
    node_names=()

    for worker in "${WORKERS[@]}"; do
        start_ray_node "$worker" false &
        pids+=($!)
        node_names+=("$worker")
    done

    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        node=${node_names[$i]}
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

# --- 7. 最终报告 ---
echo ""
echo -e "${BLUE}=============================================${NC}"
if [ $failed_count -eq 0 ]; then
    log_message "SUCCESS" "$GREEN" "Ray cluster setup completed successfully!"
else
    log_warn "Ray cluster setup completed with $failed_count worker node(s) failed!"
    log_info "Failed nodes: ${failed_nodes[*]}"
fi

log_info "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "Total nodes: $NUM_NODES ($success_count nodes running Ray)"
echo -e "${BLUE}=============================================${NC}"

echo ""
echo -e "${BLUE}=============================================${NC}"
log_info "Displaying Ray status..."
if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$MASTER_ADDR" \
    "cd '${PROJECT_DIR}' && source set_env.sh && \
    docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" \
    bash -c \"cd '${PROJECT_DIR}' && source set_env.sh && ray status\""; then
    log_warn "Failed to retrieve Ray status. Cluster may still be initializing."
fi
echo -e "${BLUE}=============================================${NC}"
