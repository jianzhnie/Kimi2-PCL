#!/usr/bin/env bash

# ==========================================
# 环境变量配置 (set_env.sh)
# 包含 Ray 集群部署、vLLM 推理及 Ascend NPU 的相关配置
# ==========================================

# ------------------------------------------
# 1. 部署与节点配置
# ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NODES_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
export MASTER_NODE="${MASTER_NODE:-}"

export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------
export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"
export VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-${SCRIPT_DIR}/vllm_model_server.sh}"

# ------------------------------------------
# 3. Ray 与 vLLM 配置
# ------------------------------------------
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8000}"

# ------------------------------------------
# 4. Ascend NPU 与底层环境配置
# ------------------------------------------
# 注意: 下列 source 命令通常在容器内生效
# 由于第三方脚本（如 Ascend 的 set_env.sh）可能存在未绑定变量，临时关闭 set -u 检查
set +u

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
#     source /usr/local/Ascend/nnal/atb/set_env.sh
# fi

source /llm_workspace_1P/expert_monitor/ATB/ascend-transformer-boost-master/output/atb/set_env.sh

# 恢复 set -u 检查 (如果原来是开启状态)
# 但通常 set_env.sh 被 source 时，我们确保接下来的脚本继续保持严格模式
set -u

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=enp66s0f0
export HCCL_SOCKET_IFNAME=enp66s0f0
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1

# 自动获取业务网卡 IP 并统一绑定，确保 Ray 和 vLLM 互相识别的节点 IP 完全一致
if command -v ip >/dev/null 2>&1; then
    _NODE_IP=$(ip -4 addr show ${GLOO_SOCKET_IFNAME} 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1)
    if [ -n "$_NODE_IP" ]; then
        export RAY_NODE_IP_ADDRESS="$_NODE_IP"
        export VLLM_HOST_IP="$_NODE_IP"
    fi
fi
