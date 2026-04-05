#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------
# 容器与镜像配置
# ------------------------------------------
export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-/llm_workspace_1P/robin/Kimi2-PCL/vllm-infer/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"

# ------------------------------------------
# 网络及Ascend配置
# ------------------------------------------
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp66s0f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f0}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

# Ray 启动超时配置
export RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-120}"
export RAY_CONNECT_TIMEOUT="${RAY_CONNECT_TIMEOUT:-60}"

# ------------------------------------------
# Ray 配置
# ------------------------------------------
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
export DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WAIT_TIME="${WAIT_TIME:-1}"

# ------------------------------------------
# vLLM 多节点网络配置
# ------------------------------------------
# VLLM_HOST_IP 必须与 Ray 集群节点 IP 一致，否则 placement group 创建会失败
# 默认使用当前主机 IP，多节点环境下需显式设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_LIST_FILE="${SCRIPT_DIR}/node_list.txt"
if [[ -f "$NODE_LIST_FILE" ]]; then
    # 使用 node_list.txt 的第一个节点作为 VLLM_HOST_IP
    VLLM_HOST_IP_DEFAULT=$(head -1 "$NODE_LIST_FILE" | awk '{print $1}')
fi
export VLLM_HOST_IP="${VLLM_HOST_IP:-VLLM_HOST_IP_DEFAULT}"
export VLLM_PORT="${VLLM_PORT:-29501}"

# ------------------------------------------
# Ascend NPU 与底层环境配置
# ------------------------------------------
# 注意: 下列 source 命令通常在容器内生效
# 由于第三方脚本（如 Ascend 的 set_env.sh）可能存在未绑定变量，临时关闭 set -u 检查
set +u

# 加载 Ascend Toolkit 环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 加载 ATB 环境（如果存在）
# 尝试多个可能的路径
_ATB_PATHS=(
    "/usr/local/Ascend/nnal/atb/set_env.sh"
    "/llm_workspace_1P/expert_monitor/ATB/ascend-transformer-boost-master/output/atb/set_env.sh"
    )

for _atb_path in "${_ATB_PATHS[@]}"; do
    if [ -f "$_atb_path" ]; then
        source "$_atb_path"
        break
    fi
done
