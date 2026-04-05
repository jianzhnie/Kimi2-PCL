#!/usr/bin/env bash

# =================================================================
# Ray Cluster Launcher
# =================================================================
# 启动多节点 Ray 集群，支持配置 NPU 资源
# 依赖: 所有节点上安装 Ray 并配置无密码 SSH 登录
# =================================================================

set -euo pipefail

# -----------------------------------------------------------------
# 配置与常量
# -----------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/set_env.sh"
NODE_LIST_FILE="${NODES_FILE:-${PROJECT_DIR}/node_list.txt}"

DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPUS_PER_NODE=8
WAIT_TIME=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# -----------------------------------------------------------------
# 日志函数
# -----------------------------------------------------------------
log() {
    local level=$1 color=$2 msg=$3
    echo -e "${color}[${level}]${NC} $(date '+%Y-%m-%d %H:%M:%S') - ${msg}"
}

log_info()  { log "INFO"  "$GREEN"  "$1"; }
log_warn()  { log "WARN"  "$YELLOW" "$1" >&2; }
log_error() { log "ERROR" "$RED"    "$1" >&2; }
log_fatal() { log "FATAL" "$RED"    "$1" >&2; exit 1; }

# -----------------------------------------------------------------
# 远程执行辅助函数
# -----------------------------------------------------------------
ssh_cmd() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@"
}

remote_exec() {
    local node=$1 cmd=$2
    ssh_cmd "$node" "cd '${PROJECT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -c '${cmd}'"
}

# -----------------------------------------------------------------
# Ray 操作函数
# -----------------------------------------------------------------
stop_ray() {
    remote_exec "$1" "ray stop -f 2>/dev/null || true"
}

start_head() {
    local node=$1 master_addr=$2
    log_info "[HEAD] Starting Ray head on $node"
    
    local cmd="ray start --head \
        --node-ip-address=${master_addr} \
        --port ${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --num-gpus=${NPUS_PER_NODE}"
    
    remote_exec "$node" "$cmd"
}

start_worker() {
    local node=$1 master_addr=$2
    log_info "[WORKER] Starting Ray worker on $node"
    
    local cmd="ray start \
        --address ${master_addr}:${MASTER_PORT} \
        --node-ip-address=${node} \
        --num-gpus=${NPUS_PER_NODE}"
    
    remote_exec "$node" "$cmd"
}

# -----------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------
check_env() {
    [[ -f "$ENV_FILE" ]] || log_fatal "Environment file not found: $ENV_FILE"
    source "$ENV_FILE"
}

check_node() {
    local node=$1
    ssh_cmd -q "$node" "exit" 2>/dev/null || { log_error "SSH failed: $node"; return 1; }
    ssh_cmd "$node" "[ -d '$PROJECT_DIR' ]" 2>/dev/null || { log_error "Directory not found on $node: $PROJECT_DIR"; return 1; }
    ssh_cmd "$node" "[ -f '$PROJECT_DIR/set_env.sh' ]" 2>/dev/null || { log_error "set_env.sh not found on $node"; return 1; }
    return 0
}

# -----------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------
main() {
    check_env
    
    # 解析节点列表
    [[ -f "$NODE_LIST_FILE" ]] || log_fatal "Node list file not found: $NODE_LIST_FILE"
    mapfile -t NODES < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")
    [[ ${#NODES[@]} -gt 0 ]] || log_fatal "No valid hosts in: $NODE_LIST_FILE"
    
    local master_addr="${MASTER_ADDR:-${NODES[0]}}"
    local num_nodes=${#NODES[@]}
    
    # 分离 worker 节点
    local workers=()
    for node in "${NODES[@]}"; do
        [[ "$node" != "$master_addr" ]] && workers+=("$node")
    done
    
    # 打印配置
    log_info "============================================="
    log_info "Ray Cluster Configuration"
    log_info "============================================="
    log_info "Total nodes: $num_nodes"
    log_info "NPUs per node: $NPUS_PER_NODE"
    log_info "Master: ${BLUE}${master_addr}${NC}:${MASTER_PORT}"
    log_info "Dashboard: http://${master_addr}:${DASHBOARD_PORT}"
    log_info "Workers (${#workers[@]}): ${workers[*]:-None}"
    log_info "============================================="
    
    # 节点检查
    log_info "Checking all nodes..."
    local errors=0
    for node in "${NODES[@]}"; do
        check_node "$node" || ((errors++))
    done
    [[ $errors -eq 0 ]] || log_fatal "Pre-checks failed with $errors error(s)"
    log_info "All checks passed"
    
    # 启动 Head 节点
    stop_ray "$master_addr"
    start_head "$master_addr" "$master_addr" || log_fatal "Failed to start head node"
    sleep "$WAIT_TIME"
    
    # 启动 Worker 节点（并行）
    local success=1 failed=0
    local failed_nodes=()
    
    if [[ ${#workers[@]} -gt 0 ]]; then
        log_info "Starting ${#workers[@]} worker(s) in parallel..."
        local pids=()
        for node in "${workers[@]}"; do
            stop_ray "$node"
            (
                start_worker "$node" "$master_addr" && \
                    log_info "Worker connected: $node"
            ) || { failed_nodes+=("$node"); ((failed++)); }
            pids+=($!)
        done
        
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        success=$((num_nodes - failed))
    else
        log_info "Single-node cluster mode"
    fi
    
    # 结果报告
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    if [[ $failed -eq 0 ]]; then
        log "SUCCESS" "$GREEN" "Ray cluster started successfully!"
    else
        log_warn "Cluster started with $failed failed worker(s)"
        log_info "Failed nodes: ${failed_nodes[*]}"
    fi
    log_info "Dashboard: http://${master_addr}:${DASHBOARD_PORT}"
    log_info "Running: $success / $num_nodes nodes"
    echo -e "${BLUE}=============================================${NC}"
    
    # 显示状态
    log_info "Ray cluster status:"
    ssh_cmd "$master_addr" "cd '${PROJECT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" \
        bash -c 'ray status'" 2>/dev/null || log_warn "Could not retrieve Ray status"
}

main "$@"
