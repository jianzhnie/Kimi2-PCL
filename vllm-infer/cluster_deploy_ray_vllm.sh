#!/usr/bin/env bash
# ==========================================
# 集群部署脚本 (cluster_deploy_ray_vllm.sh)
# 用于在多节点集群上部署容器，启动 Ray 集群以及 vLLM 服务。
# ==========================================

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

# ------------------------------------------
# 日志函数
# ------------------------------------------
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_err()  { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash cluster_deploy_ray_vllm.sh [--prepare-only|--ray-only|--serve-only]

Description:
  该脚本用于在集群节点上拉起容器环境，并部署 Ray 集群及 vLLM 模型服务。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  --prepare-only   仅在各节点准备 Docker 容器
  --ray-only       仅启动 Ray 集群 (包括 head 和 worker 节点)
  --serve-only     仅在 Master 节点上启动 vLLM 模型服务
  -h, --help       显示帮助信息
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
MODE="all"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--prepare-only" ]]; then
  MODE="prepare"
elif [[ "${1:-}" == "--ray-only" ]]; then
  MODE="ray"
elif [[ "${1:-}" == "--serve-only" ]]; then
  MODE="serve"
elif [[ -n "${1:-}" ]]; then
  log_err "未知选项: $1"
  usage >&2
  exit 2
fi

# ------------------------------------------
# 辅助函数
# ------------------------------------------

# 获取非空节点列表
read_nodes() {
  if [[ ! -f "$NODES_FILE" ]]; then
    log_err "节点列表文件未找到: $NODES_FILE"
    exit 2
  fi
  awk 'NF {print $1}' "$NODES_FILE"
}

# 拼接 SSH 目标地址
ssh_target() {
  local node="$1"
  printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

# 执行 SSH 命令
ssh_run() {
  local node="$1"
  shift
  ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n
  done
}

# 推断主节点
detect_master() {
  if [[ -n "$MASTER_NODE" ]]; then
    echo "$MASTER_NODE"
    return 0
  fi
  read_nodes | head -n 1
}

# ------------------------------------------
# 容器内远程执行逻辑定义
# 注意: 下列带 _remote 前缀的函数将通过 declare -f 被序列化发送至远程执行
# ------------------------------------------

_remote_prepare_node() {
  local image_name="$1"
  local image_tar="$2"
  local run_container_script="$3"
  local container_name="$4"

  set -euo pipefail
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker not found" >&2
    exit 127
  fi

  # 确保 Docker 服务已启动
  echo "[INFO] Starting Docker service..."
  systemctl daemon-reload && systemctl start docker || {
    echo "[WARN] Failed to start Docker via systemctl, trying to check Docker status..." >&2
  }

  if docker image inspect "${image_name}" >/dev/null 2>&1; then
    : # 镜像已存在
  else
    if [ ! -f "${image_tar}" ]; then
      echo "[ERROR] image tar not found: ${image_tar}" >&2
      exit 2
    fi
    echo "[INFO] Loading image from ${image_tar}..."
    docker load -i "${image_tar}"
  fi

  if [ ! -f "${run_container_script}" ]; then
    echo "[ERROR] run script not found: ${run_container_script}" >&2
    exit 2
  fi

  export IMAGE_NAME="${image_name}"
  export CONTAINER_NAME="${container_name}"
  bash "${run_container_script}"

  if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
    echo "[INFO] Container ready: ${container_name}"
  else
    echo "[ERROR] Failed to start container: ${container_name}" >&2
    exit 1
  fi
}

_remote_stop_ray() {
  set -euo pipefail
  ray stop -f >/dev/null 2>&1 || true
}

_remote_start_ray_head() {
  local ray_port="$1"
  local dashboard_port="$2"
  local npus="$3"

  set -euo pipefail

  local local_ip
  local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "$local_ip" ]]; then
    local_ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
  fi

  # 注意：由于 Ray 内部规定 'GPU' 是内置资源，不能放在 --resources (自定义资源) 字典里。
  # 所以我们在 --resources 中只放 NPU，通过 --num-gpus 来满足 vLLM 的 GPU 调度需求。
  local resources_json="{\"NPU\": ${npus}}"

  ray start --head \
      --port="${ray_port}" \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="${dashboard_port}" \
      --num-gpus="${npus}" \
      --resources="${resources_json}" >/dev/null 2>&1

  # 返回 IP 供外部调用
  echo "$local_ip"
}

_remote_start_ray_worker() {
  local head_ip="$1"
  local ray_port="$2"
  local npus="$3"

  set -euo pipefail

  local local_ip
  local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "$local_ip" ]]; then
    local_ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
  fi

  # 同样地，给 Worker 节点声明 NPU，并通过 --num-gpus 满足 GPU 请求
  local resources_json="{\"NPU\": ${npus}}"

  ray start --address="${head_ip}:${ray_port}" \
      --num-gpus="${npus}" \
      --resources="${resources_json}"
}

_remote_serve_vllm() {
  local vllm_start_script="$1"
  local vllm_host="$2"
  local vllm_port="$3"

  set -euo pipefail
  if [ ! -f "${vllm_start_script}" ]; then
    echo "[ERROR] vllm start script not found: ${vllm_start_script}" >&2
    exit 2
  fi

  export HOST="${vllm_host}"
  export PORT="${vllm_port}"

  echo "[INFO] Starting vLLM service..."
  nohup bash "${vllm_start_script}" >/tmp/vllm_serve.log 2>&1 &
  echo $! >/tmp/vllm_serve.pid
  echo "[INFO] vLLM service started with PID $(cat /tmp/vllm_serve.pid)"
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

# 包装器：将本地函数发送到远端执行，并预先 source 环境配置，在容器内执行
remote_exec_in_container() {
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

  # 宿主机环境 source
  local ssh_cmd="cd '${SCRIPT_DIR}' && source set_env.sh && \
      docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

  # 容器内环境 source + 执行函数
  echo "cd '${SCRIPT_DIR}' && source set_env.sh 2>/dev/null || true; ${func_code}; ${call_code}" \
      | ssh_run "$node" "$ssh_cmd"
}

prepare_node() {
  local node="$1"
  log_info "[${node}] 开始环境准备..."

  local func_code call_code
  func_code="$(declare -f _remote_prepare_node)"
  call_code="_remote_prepare_node '${IMAGE_NAME}' '${IMAGE_TAR}' '${RUN_CONTAINER_SCRIPT}' '${CONTAINER_NAME}'"

  # prepare_node 在宿主机执行，不需要进容器
  if ! echo "${func_code}; ${call_code}" | ssh_run "$node" bash -lc "bash -s"; then
     log_err "[${node}] 环境准备失败"
     return 1
  fi
  log_info "[${node}] 环境准备完成"
}

stop_ray_node() {
  local node=$1
  log_info "[${node}] 停止旧的 Ray 进程..."
  remote_exec_in_container "$node" _remote_stop_ray || true
}

start_ray_head() {
  local node="$1"
  log_info "[${node}] 正在启动 Ray Head 节点..."

  stop_ray_node "$node"

  # 注意：由于我们要获取返回的 IP，我们将输出通过 tail -n 1 截取
  # 传入 dashboard port 和 npus_per_node (假设为8, 后续可从 set_env 中取)
  local npus="${NPUS_PER_NODE:-8}"
  local dash_port="${RAY_DASHBOARD_PORT:-8266}"

  remote_exec_in_container "$node" _remote_start_ray_head \
      "${RAY_PORT}" "${dash_port}" "${npus}" \
      | tail -n 1
}

start_ray_worker() {
  local node="$1"
  local head_ip="$2"
  log_info "[${node}] 正在加入 Ray 集群 (连接至 ${head_ip})..."

  stop_ray_node "$node"

  local npus="${NPUS_PER_NODE:-8}"

  if ! remote_exec_in_container "$node" _remote_start_ray_worker \
      "${head_ip}" "${RAY_PORT}" "${npus}"; then
    log_err "[${node}] 加入 Ray 集群失败"
    return 1
  fi
  log_info "[${node}] 已成功加入 Ray 集群"
}

serve_vllm_on_master() {
  local node="$1"
  log_info "[${node}] 正在启动 vLLM 服务..."

  if ! remote_exec_in_container "$node" _remote_serve_vllm \
      "${VLLM_START_SCRIPT}" "${VLLM_HOST}" "${VLLM_PORT}"; then
    log_err "[${node}] 启动 vLLM 服务失败"
    return 1
  fi
  log_info "vLLM 服务日志位于 ${node}:/tmp/vllm_serve.log"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes)"
if [[ -z "$nodes" ]]; then
  log_err "NODES_FILE 中未找到任何节点信息"
  exit 2
fi

master="$(detect_master)"
if [[ -z "$master" ]]; then
  log_err "MASTER_NODE 未设置，且无法从 NODES_FILE 推断主节点"
  exit 2
fi

log_info "目标节点列表: $(echo $nodes | tr '\n' ' ')"
log_info "主节点: $master"
log_info "执行模式: $MODE"

# 1. 准备阶段
if [[ "$MODE" == "prepare" || "$MODE" == "all" ]]; then
  log_info "=== 开始准备节点 ==="
  for node in $nodes; do
    limit_jobs "$PARALLELISM"
    (prepare_node "$node") &
  done
  wait
  log_info "=== 节点准备完成 ==="
fi

# 2. 部署 Ray 集群阶段
if [[ "$MODE" == "ray" || "$MODE" == "all" ]]; then
  log_info "=== 开始部署 Ray 集群 ==="
  head_ip="$(start_ray_head "$master" | tail -n 1)"
  if [[ -z "$head_ip" ]]; then
    log_err "无法检测到 Master 节点($master)的 IP，Ray 启动失败"
    exit 1
  fi
  log_info "Ray Head IP 为: $head_ip"

  # 加入等待头节点初始化的逻辑，提高健壮性
  local wait_time="${WAIT_TIME:-3}"
  log_info "等待 ${wait_time}s 以确保 Head 节点初始化完成..."
  sleep "$wait_time"

  for node in $nodes; do
    if [[ "$node" == "$master" ]]; then
      continue
    fi
    limit_jobs "$PARALLELISM"
    (start_ray_worker "$node" "$head_ip") &
  done
  wait

  log_info "正在验证 Ray 集群状态..."
  local ssh_cmd="cd '${SCRIPT_DIR}' && source set_env.sh && \
      docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" \
      bash -c \"cd '${SCRIPT_DIR}' && source set_env.sh && ray status\""
  ssh_run "$master" "$ssh_cmd" || log_warn "获取 Ray 状态失败，集群可能仍在初始化中"
  log_info "=== Ray 集群部署完成 ==="
fi

# 3. 启动 vLLM 服务阶段
if [[ "$MODE" == "serve" || "$MODE" == "all" ]]; then
  log_info "=== 开始启动 vLLM 服务 ==="
  serve_vllm_on_master "$master"
  log_info "=== vLLM 服务启动操作完成 ==="
fi

log_info "任务执行完毕"
