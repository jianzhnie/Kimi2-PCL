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
    local level=$1 color=$2
    shift 2
    echo -e "${color}[${level}]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_info()  { log "INFO"  "$GREEN"  "$@"; }
log_warn()  { log "WARN"  "$YELLOW" "$@" >&2; }
log_error() { log "ERROR" "$RED"    "$@" >&2; }
log_fatal() { log "FATAL" "$RED"    "$@" >&2; exit 1; }

# -----------------------------------------------------------------
# 远程执行辅助函数
# -----------------------------------------------------------------
ssh_cmd() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$@"
}

remote_exec() {
    local node=$1 cmd=$2
    ssh_cmd "$node" "cd '${PROJECT_DIR}' && source set_env.sh 2>/dev/null && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -c '${cmd}'"
}

# -----------------------------------------------------------------
# Ray 操作函数
# -----------------------------------------------------------------

# 停止单个节点的 Ray 进程
stop_ray_on_node() {
    local node=$1
    log_info "[STOP] Stopping Ray on $node"
    remote_exec "$node" "ray stop -f 2>/dev/null || true" 2>/dev/null || true
}

# 停止所有节点的 Ray 进程（并行执行）
stop_all_ray() {
    local nodes=("$@")
    log_info "Stopping Ray on all ${#nodes[@]} node(s)..."
    
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" EXIT
    
    for i in "${!nodes[@]}"; do
        local node=${nodes[$i]}
        (
            stop_ray_on_node "$node" && echo "OK" > "$tmpdir/$i" || echo "FAIL" > "$tmpdir/$i"
        ) &
    done
    
    wait
    
    local failed=0
    for i in "${!nodes[@]}"; do
        if [[ "$(cat "$tmpdir/$i" 2>/dev/null)" != "OK" ]]; then
            log_warn "Failed to stop Ray on: ${nodes[$i]}"
            ((failed++))
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_info "All Ray processes stopped successfully"
    else
        log_warn "$failed node(s) failed to stop (may not be running)"
    fi
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
    ssh_cmd -q "$node" "test -d '$PROJECT_DIR' -a -f '$PROJECT_DIR/set_env.sh'" 2>/dev/null || {
        log_error "SSH failed or missing files on: $node"
        return 1
    }
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
    local failed=0
    for node in "${NODES[@]}"; do
        check_node "$node" || ((failed++))
    done
    [[ $failed -eq 0 ]] || log_fatal "Pre-checks failed with $failed error(s)"
    log_info "All checks passed"
    
    # Step 1: 先停止所有节点的 Ray 进程
    log_info "============================================="
    stop_all_ray "${NODES[@]}"
    sleep 1
    stop_all_ray "${NODES[@]}"

    # Step 2: 启动 Head 节点
    log_info "============================================="
    start_head "$master_addr" "$master_addr" || log_fatal "Failed to start head node"
    sleep "$WAIT_TIME"
    
    # 启动 Worker 节点（并行）
    local failed_nodes=() success=$((num_nodes))
    
    if [[ ${#workers[@]} -gt 0 ]]; then
        log_info "Starting ${#workers[@]} worker(s) in parallel..."
        
        # 并行启动 workers，将结果写入临时文件
        local tmpdir=$(mktemp -d)
        trap "rm -rf $tmpdir" EXIT
        
        for i in "${!workers[@]}"; do
            local node=${workers[$i]}
            (
                start_worker "$node" "$master_addr" && echo "OK" > "$tmpdir/$i" || echo "FAIL" > "$tmpdir/$i"
            ) &
        done
        
        # 等待所有后台任务完成
        wait
        
        # 收集结果
        for i in "${!workers[@]}"; do
            if [[ "$(cat "$tmpdir/$i" 2>/dev/null)" != "OK" ]]; then
                failed_nodes+=("${workers[$i]}")
                ((success--))
            fi
        done
    else
        log_info "Single-node cluster mode"
    fi
    
    # 结果报告
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    if [[ ${#failed_nodes[@]} -eq 0 ]]; then
        log "SUCCESS" "$GREEN" "Ray cluster started successfully!"
    else
        log_warn "Cluster started with ${#failed_nodes[@]} failed worker(s)"
        log_info "Failed nodes: ${failed_nodes[*]}"
    fi
    log_info "Dashboard: http://${master_addr}:${DASHBOARD_PORT}"
    log_info "Running: $success / $num_nodes nodes"
    echo -e "${BLUE}=============================================${NC}"
    
    # 显示状态
    log_info "Ray cluster status:"
    ssh_cmd "$master_addr" "cd '${PROJECT_DIR}' && source set_env.sh 2>/dev/null && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" \
        bash -c 'ray status'" 2>/dev/null || log_warn "Could not retrieve Ray status"
}

main "$@"
