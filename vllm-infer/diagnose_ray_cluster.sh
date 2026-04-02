#!/usr/bin/env bash
# =================================================================
# Ray Cluster 诊断工具
# 用途: 检查节点间通信和 IP 配置一致性
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/set_env.sh" 2>/dev/null || {
    echo "[ERROR] 无法加载 set_env.sh"
    exit 1
}

NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# 获取节点列表
mapfile -t NODE_HOSTS < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")
MASTER_ADDR="${MASTER_NODE:-${NODE_HOSTS[0]}}"

echo "============================================="
echo "Ray Cluster 诊断报告"
echo "============================================="
echo "Head 节点: $MASTER_ADDR"
echo "总节点数: ${#NODE_HOSTS[@]}"
echo "============================================="
echo ""

# 1. 检查每个节点的 IP 配置
echo "【1. 节点 IP 配置检查】"
echo "---------------------------------------------"

for node in "${NODE_HOSTS[@]}"; do
    result=$(ssh $SSH_OPTS "$node" '
        cd "'"$SCRIPT_DIR"'" && source set_env.sh 2>/dev/null
        echo "HOSTNAME:$(hostname)"
        echo "RAY_NODE_IP_ADDRESS:${RAY_NODE_IP_ADDRESS:-NOT_SET}"
        echo "VLLM_HOST_IP:${VLLM_HOST_IP:-NOT_SET}"
        echo "GLOO_SOCKET_IFNAME:${GLOO_SOCKET_IFNAME:-NOT_SET}"
        # 获取网卡 IP
        if [ -n "${GLOO_SOCKET_IFNAME:-}" ]; then
            ip addr show "${GLOO_SOCKET_IFNAME}" 2>/dev/null | awk "/inet / {print \"IFACE_IP:\" \$2}" | cut -d/ -f1
        fi
        # 获取默认路由 IP
        ip route get 1.1.1.1 2>/dev/null | awk "/src/ {for(i=1;i<=NF;i++) if(\$i==\"src\") print \"DEFAULT_IP:\" \$(i+1)}"
    ' 2>/dev/null || echo "SSH_FAILED")
    
    echo "节点: $node"
    if [[ "$result" == "SSH_FAILED" ]]; then
        log_error "SSH 连接失败"
    else
        echo "$result" | while read -r line; do
            case "$line" in
                HOSTNAME:*) echo "  主机名: ${line#HOSTNAME:}" ;;
                RAY_NODE_IP_ADDRESS:*) 
                    ip="${line#RAY_NODE_IP_ADDRESS:}"
                    if [[ "$ip" == "NOT_SET" ]]; then
                        log_error "  RAY_NODE_IP_ADDRESS: 未设置!"
                    else
                        echo "  RAY_NODE_IP_ADDRESS: $ip"
                    fi
                    ;;
                VLLM_HOST_IP:*) 
                    ip="${line#VLLM_HOST_IP:}"
                    if [[ "$ip" == "NOT_SET" ]]; then
                        log_error "  VLLM_HOST_IP: 未设置!"
                    else
                        echo "  VLLM_HOST_IP: $ip"
                    fi
                    ;;
                IFACE_IP:*) echo "  网卡 IP: ${line#IFACE_IP:}" ;;
                DEFAULT_IP:*) echo "  默认路由 IP: ${line#DEFAULT_IP:}" ;;
            esac
        done
    fi
    echo ""
done

# 2. 检查 Ray 状态
echo "【2. Ray 集群状态检查】"
echo "---------------------------------------------"

ssh $SSH_OPTS "$MASTER_ADDR" '
    cd "'"$SCRIPT_DIR"'" && source set_env.sh
    docker exec "${CONTAINER_NAME:-vllm-ascend-env-a3}" bash -c "
        echo \"--- Ray 状态 ---\"
        ray status
        echo \""
        echo \"--- Ray 节点列表 ---\"
        ray list nodes 2>/dev/null || echo \"无法获取节点列表\"
    "
' 2>/dev/null || log_error "无法获取 Ray 状态"

echo ""

# 3. 检查节点间网络连通性
echo "【3. 节点间网络连通性检查】"
echo "---------------------------------------------"

# 获取 head 节点的 IP
HEAD_IP=$(ssh $SSH_OPTS "$MASTER_ADDR" '
    source "'"$SCRIPT_DIR"'"/set_env.sh 2>/dev/null
    echo "${RAY_NODE_IP_ADDRESS:-$(hostname -I | awk "{print \$1}")}"
')

for node in "${NODE_HOSTS[@]}"; do
    if [[ "$node" == "$MASTER_ADDR" ]]; then
        continue
    fi
    
    # 从 worker 节点 ping head 节点
    result=$(ssh $SSH_OPTS "$node" "
        ping -c 1 -W 2 $HEAD_IP >/dev/null 2>&1 && echo \"REACHABLE\" || echo \"UNREACHABLE\"
    " 2>/dev/null)
    
    if [[ "$result" == "REACHABLE" ]]; then
        log_info "Worker $node -> Head $HEAD_IP: 可连通"
    else
        log_error "Worker $node -> Head $HEAD_IP: 无法连通!"
    fi
done

echo ""

# 4. 检查 Ray 端口连通性
echo "【4. Ray 端口连通性检查】"
echo "---------------------------------------------"

RAY_PORT="${RAY_PORT:-6379}"

for node in "${NODE_HOSTS[@]}"; do
    if [[ "$node" == "$MASTER_ADDR" ]]; then
        continue
    fi
    
    # 检查 GCS 端口
    result=$(ssh $SSH_OPTS "$node" "
        timeout 2 bash -c \"cat < /dev/null > /dev/tcp/$MASTER_ADDR/$RAY_PORT\" 2>/dev/null && echo \"OPEN\" || echo \"CLOSED\"
    " 2>/dev/null)
    
    if [[ "$result" == "OPEN" ]]; then
        log_info "Worker $node -> Head:$RAY_PORT: 端口开放"
    else
        log_error "Worker $node -> Head:$RAY_PORT: 端口未开放!"
    fi
done

echo ""

# 5. 检查防火墙
echo "【5. 防火墙状态检查】"
echo "---------------------------------------------"

for node in "${NODE_HOSTS[@]}"; do
    firewall=$(ssh $SSH_OPTS "$node" '
        if command -v ufw >/dev/null 2>&1; then
            ufw status 2>/dev/null | grep -q "Status: active" && echo "UFW_ACTIVE" || echo "UFW_INACTIVE"
        elif command -v firewall-cmd >/dev/null 2>&1; then
            firewall-cmd --state 2>/dev/null | grep -q "running" && echo "FIREWALLD_ACTIVE" || echo "FIREWALLD_INACTIVE"
        elif command -v iptables >/dev/null 2>&1; then
            iptables -L 2>/dev/null | grep -q "DROP" && echo "IPTABLES_ACTIVE" || echo "IPTABLES_INACTIVE"
        else
            echo "UNKNOWN"
        fi
    ' 2>/dev/null)
    
    case "$firewall" in
        *ACTIVE) log_warn "节点 $node: 防火墙正在运行 ($firewall)" ;;
        *) log_info "节点 $node: 防火墙未激活 ($firewall)" ;;
    esac
done

echo ""
echo "============================================="
echo "诊断完成"
echo "============================================="
